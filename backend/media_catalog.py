"""
Media Catalog — SQLite-backed generation asset management for ComfyUI.

Every image generation is recorded with full metadata (workflow, params, files)
for search, reproduction, lineage tracking, and gallery browsing.
"""

import json
import logging
import os
import shutil
import sqlite3
import struct
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlretrieve

logger = logging.getLogger("media_catalog")

# Supported file extensions by media type
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".webm", ".gif", ".mkv", ".avi", ".mov"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
ALL_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS


def detect_media_type(filename: str) -> str:
    """Detect media type from file extension. Returns 'image', 'video', 'audio', or 'unknown'."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return "unknown"


class MediaCatalog:
    """
    Generation asset catalog backed by SQLite.

    Thread-safe: opens/closes connections per operation.
    All public methods are sync (matching n8n_db_utils.py pattern).
    """

    def __init__(self, db_path: str, comfyui_output_dir: str = None):
        self.db_path = db_path
        self.output_dir = comfyui_output_dir
        self._read_conn: sqlite3.Connection = None  # Persistent read connection
        self._init_db()
        self._migrate_db()
        # TTL cache for get_stats()
        self._stats_cache = None
        self._stats_cache_time = 0.0
        self._STATS_CACHE_TTL = 5.0  # seconds

    # ================================================================
    # Database lifecycle
    # ================================================================

    def _init_db(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = self._conn()
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS generations (
                id              TEXT PRIMARY KEY,
                prompt_id       TEXT NOT NULL,
                instance_id     TEXT,
                instance_port   INTEGER,
                parent_id       TEXT,
                workflow_type   TEXT,
                status          TEXT DEFAULT 'completed',
                checkpoint      TEXT,
                prompt_text     TEXT,
                negative_prompt TEXT,
                seed            INTEGER,
                steps           INTEGER,
                cfg             REAL,
                sampler         TEXT,
                scheduler       TEXT,
                denoise         REAL,
                width           INTEGER,
                height          INTEGER,
                workflow_json   TEXT,
                title           TEXT,
                tags            TEXT DEFAULT '',
                rating          INTEGER DEFAULT 0,
                favorite        INTEGER DEFAULT 0,
                notes           TEXT,
                created_at      TEXT DEFAULT (datetime('now')),
                updated_at      TEXT DEFAULT (datetime('now')),
                comfyui_dir     TEXT,
                FOREIGN KEY (parent_id) REFERENCES generations(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS generation_files (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id   TEXT NOT NULL,
                filename        TEXT NOT NULL,
                subfolder       TEXT DEFAULT '',
                file_type       TEXT DEFAULT 'output',
                disk_path       TEXT,
                file_size       INTEGER,
                width           INTEGER,
                height          INTEGER,
                format          TEXT DEFAULT 'png',
                media_type      TEXT DEFAULT 'image',
                duration        REAL,
                is_primary      INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (generation_id) REFERENCES generations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS orphan_files (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT NOT NULL UNIQUE,
                subfolder       TEXT DEFAULT '',
                disk_path       TEXT,
                file_size       INTEGER,
                width           INTEGER,
                height          INTEGER,
                format          TEXT DEFAULT 'png',
                media_type      TEXT DEFAULT 'image',
                duration        REAL,
                discovered_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_gen_created ON generations(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_gen_checkpoint ON generations(checkpoint);
            CREATE INDEX IF NOT EXISTS idx_gen_prompt ON generations(prompt_text);
            CREATE INDEX IF NOT EXISTS idx_gen_favorite ON generations(favorite);
            CREATE INDEX IF NOT EXISTS idx_gen_status ON generations(status);
            CREATE INDEX IF NOT EXISTS idx_gen_parent ON generations(parent_id);
            CREATE INDEX IF NOT EXISTS idx_gen_prompt_id ON generations(prompt_id);
            CREATE INDEX IF NOT EXISTS idx_genfile_genid ON generation_files(generation_id);
            CREATE INDEX IF NOT EXISTS idx_genfile_filename ON generation_files(filename);
        """)
        conn.commit()
        conn.close()
        logger.info(f"Media catalog initialized at {self.db_path}")

    def _migrate_db(self):
        """Add columns that may be missing in databases created before media_type support."""
        conn = self._conn()
        try:
            for table in ("generation_files", "orphan_files"):
                cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                if "media_type" not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN media_type TEXT DEFAULT 'image'")
                    logger.info(f"Migrated {table}: added media_type column")
                if "duration" not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN duration REAL")
                    logger.info(f"Migrated {table}: added duration column")
            conn.commit()
        except Exception as e:
            logger.warning(f"Migration check failed (non-fatal): {e}")
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        """Create a new connection for write operations."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _read(self) -> sqlite3.Connection:
        """Get a persistent read connection (fast, no per-call PRAGMA overhead)."""
        if self._read_conn is None:
            self._read_conn = sqlite3.connect(self.db_path)
            self._read_conn.row_factory = sqlite3.Row
            self._read_conn.execute("PRAGMA journal_mode=WAL")
            self._read_conn.execute("PRAGMA foreign_keys=ON")
            self._read_conn.execute("PRAGMA cache_size=-4000")  # 4MB page cache
            self._read_conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            self._read_conn.execute("PRAGMA temp_store=MEMORY")
        return self._read_conn

    # ================================================================
    # Core CRUD
    # ================================================================

    def record_generation(
        self,
        prompt_id: str,
        instance_id: str,
        instance_port: int,
        workflow_json: dict,
        images: List[dict],
        parent_id: str = None,
        comfyui_dir: str = None,
    ) -> str:
        """Record a completed generation. Returns generation_id."""
        gen_id = str(uuid.uuid4())
        params = self.extract_params(workflow_json) if workflow_json else {}
        output_dir = comfyui_dir or self.output_dir or ""

        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO generations
                   (id, prompt_id, instance_id, instance_port, parent_id,
                    workflow_type, checkpoint, prompt_text, negative_prompt,
                    seed, steps, cfg, sampler, scheduler, denoise, width, height,
                    workflow_json, comfyui_dir)
                   VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?,?,?,?,?, ?,?)""",
                (gen_id, prompt_id, instance_id, instance_port, parent_id,
                 params.get("workflow_type", "custom"),
                 params.get("checkpoint"), params.get("prompt_text"),
                 params.get("negative_prompt"),
                 params.get("seed"), params.get("steps"), params.get("cfg"),
                 params.get("sampler"), params.get("scheduler"),
                 params.get("denoise"), params.get("width"), params.get("height"),
                 json.dumps(workflow_json) if workflow_json else None,
                 output_dir),
            )

            for i, img in enumerate(images):
                filename = img.get("filename", "")
                subfolder = img.get("subfolder", "")
                disk_path = img.get("disk_path", "")

                if not disk_path and output_dir and filename:
                    if subfolder:
                        disk_path = os.path.join(output_dir, subfolder, filename)
                    else:
                        disk_path = os.path.join(output_dir, filename)

                file_info = self.measure_file(disk_path) if disk_path and os.path.exists(disk_path) else {}
                fmt = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
                mt = file_info.get("media_type", detect_media_type(filename))

                conn.execute(
                    """INSERT INTO generation_files
                       (generation_id, filename, subfolder, file_type, disk_path,
                        file_size, width, height, format, media_type, duration, is_primary)
                       VALUES (?,?,?,?,?, ?,?,?,?,?,?,?)""",
                    (gen_id, filename, subfolder,
                     img.get("type", "output"), disk_path,
                     file_info.get("file_size"), file_info.get("width"),
                     file_info.get("height"), fmt, mt, file_info.get("duration"),
                     1 if i == 0 else 0),
                )

            conn.commit()
            logger.info(f"Cataloged generation {gen_id} ({len(images)} files) for prompt {prompt_id}")
            return gen_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to catalog generation: {e}")
            raise
        finally:
            conn.close()

    def get_generation(self, generation_id: str) -> Optional[dict]:
        """Get generation with its files."""
        conn = self._read()
        row = conn.execute(
            "SELECT * FROM generations WHERE id = ?", (generation_id,)
        ).fetchone()
        if not row:
            return None

        gen = dict(row)
        if gen.get("workflow_json"):
            try:
                gen["workflow_json"] = json.loads(gen["workflow_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        files = conn.execute(
            "SELECT * FROM generation_files WHERE generation_id = ? ORDER BY is_primary DESC, id",
            (generation_id,)
        ).fetchall()
        gen["files"] = [dict(f) for f in files]
        return gen

    def get_generation_by_prompt_id(self, prompt_id: str) -> Optional[dict]:
        """Get generation by ComfyUI prompt_id."""
        conn = self._read()
        row = conn.execute(
            "SELECT id FROM generations WHERE prompt_id = ?", (prompt_id,)
        ).fetchone()
        if not row:
            return None
        return self.get_generation(row["id"])

    def update_generation(self, generation_id: str, **kwargs) -> bool:
        """Update user metadata fields (title, tags, rating, favorite, notes)."""
        allowed = {"title", "tags", "rating", "favorite", "notes"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False

        updates["updated_at"] = "datetime('now')"
        set_parts = []
        values = []
        for k, v in updates.items():
            if k == "updated_at":
                set_parts.append(f"{k} = datetime('now')")
            else:
                set_parts.append(f"{k} = ?")
                values.append(v)
        values.append(generation_id)

        conn = self._conn()
        try:
            result = conn.execute(
                f"UPDATE generations SET {', '.join(set_parts)} WHERE id = ?",
                values,
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    def delete_generation(self, generation_id: str) -> bool:
        """Delete generation and cascade-delete file records (not disk files)."""
        conn = self._conn()
        try:
            result = conn.execute(
                "DELETE FROM generations WHERE id = ?", (generation_id,)
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    # ================================================================
    # Search / Query
    # ================================================================

    def search_generations(
        self,
        query: str = None,
        checkpoint: str = None,
        tags: str = None,
        favorite: bool = None,
        min_rating: int = None,
        date_from: str = None,
        date_to: str = None,
        sort: str = "newest",
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        """Search generations with flexible filters."""
        conditions = ["status != 'failed'"]
        params = []

        if query:
            conditions.append("prompt_text LIKE ?")
            params.append(f"%{query}%")
        if checkpoint:
            conditions.append("checkpoint LIKE ?")
            params.append(f"%{checkpoint}%")
        if tags:
            for tag in tags.split(","):
                tag = tag.strip()
                if tag:
                    conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")
        if favorite is not None:
            conditions.append("favorite = ?")
            params.append(1 if favorite else 0)
        if min_rating is not None:
            conditions.append("rating >= ?")
            params.append(min_rating)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)

        where = " AND ".join(conditions) if conditions else "1=1"

        sort_map = {
            "newest": "created_at DESC",
            "oldest": "created_at ASC",
            "rating": "rating DESC, created_at DESC",
            "favorites": "favorite DESC, created_at DESC",
        }
        order = sort_map.get(sort, "created_at DESC")

        conn = self._read()
        rows = conn.execute(
            f"""SELECT g.*, gf.filename as primary_filename,
                gf.subfolder as primary_subfolder,
                gf.media_type as primary_media_type,
                gf.duration as primary_duration
                FROM generations g
                LEFT JOIN generation_files gf ON gf.generation_id = g.id AND gf.is_primary = 1
                WHERE {where}
                ORDER BY {order}
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

        total = conn.execute(
            f"SELECT COUNT(*) as cnt FROM generations WHERE {where}",
            params,
        ).fetchone()["cnt"]

        results = []
        for row in rows:
            d = dict(row)
            d.pop("workflow_json", None)  # Don't include full workflow in search results
            results.append(d)

        return results, total

    def get_lineage(self, generation_id: str) -> List[dict]:
        """Walk parent chain up, then find children. Returns ordered list."""
        conn = self._read()
        # Walk up to root
        ancestors = []
        current_id = generation_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            row = conn.execute(
                "SELECT id, parent_id, prompt_text, checkpoint, workflow_type, created_at "
                "FROM generations WHERE id = ?",
                (current_id,)
            ).fetchone()
            if not row:
                break
            ancestors.insert(0, dict(row))
            current_id = row["parent_id"]

        # Find direct children of the target
        children = conn.execute(
            "SELECT id, parent_id, prompt_text, checkpoint, workflow_type, created_at "
            "FROM generations WHERE parent_id = ? ORDER BY created_at",
            (generation_id,)
        ).fetchall()

        # Combine: ancestors (ending with target) + children
        lineage = ancestors
        for child in children:
            child_dict = dict(child)
            if child_dict["id"] not in visited:
                lineage.append(child_dict)

        return lineage

    # ================================================================
    # Statistics
    # ================================================================

    def get_stats(self) -> dict:
        """Get catalog statistics (cached for 5s)."""
        import time as _time
        now = _time.time()
        if self._stats_cache and (now - self._stats_cache_time) < self._STATS_CACHE_TTL:
            return self._stats_cache
        conn = self._read()
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM generations WHERE status = 'completed'"
        ).fetchone()["cnt"]

        total_files = conn.execute(
            "SELECT COUNT(*) as cnt FROM generation_files"
        ).fetchone()["cnt"]

        total_bytes = conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) as total FROM generation_files"
        ).fetchone()["total"]

        favorites = conn.execute(
            "SELECT COUNT(*) as cnt FROM generations WHERE favorite = 1"
        ).fetchone()["cnt"]

        by_checkpoint = {}
        rows = conn.execute(
            "SELECT checkpoint, COUNT(*) as cnt FROM generations "
            "WHERE status = 'completed' AND checkpoint IS NOT NULL "
            "GROUP BY checkpoint ORDER BY cnt DESC"
        ).fetchall()
        for row in rows:
            by_checkpoint[row["checkpoint"]] = row["cnt"]

        result = {
            "total_generations": total,
            "total_files": total_files,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 1) if total_bytes else 0,
            "favorites_count": favorites,
            "by_checkpoint": by_checkpoint,
        }
        self._stats_cache = result
        self._stats_cache_time = _time.time()
        return result

    # ================================================================
    # Parameter extraction from workflow JSON
    # ================================================================

    @staticmethod
    def extract_params(workflow_json: dict) -> dict:
        """
        Extract key parameters from a ComfyUI API-format workflow.

        Walks nodes to find checkpoint, prompt, seed, steps, cfg, etc.
        """
        if not workflow_json or not isinstance(workflow_json, dict):
            return {}

        params = {}
        has_load_image = False

        for node_id, node in workflow_json.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            if class_type == "CheckpointLoaderSimple":
                params["checkpoint"] = inputs.get("ckpt_name")

            elif class_type == "UNETLoader":
                params["checkpoint"] = inputs.get("unet_name")

            elif class_type == "CLIPTextEncode":
                text = inputs.get("text", "")
                if text and "prompt_text" not in params:
                    params["prompt_text"] = text
                elif text and "negative_prompt" not in params:
                    params["negative_prompt"] = text

            elif class_type in ("KSampler", "KSamplerAdvanced"):
                params["seed"] = inputs.get("seed")
                params["steps"] = inputs.get("steps")
                params["cfg"] = inputs.get("cfg")
                params["sampler"] = inputs.get("sampler_name")
                params["scheduler"] = inputs.get("scheduler")
                params["denoise"] = inputs.get("denoise")

            elif class_type == "EmptyLatentImage":
                params["width"] = inputs.get("width")
                params["height"] = inputs.get("height")

            elif class_type == "LoadImage":
                has_load_image = True

            elif class_type == "UpscaleModelLoader":
                params.setdefault("workflow_type", "upscale")

            elif class_type in ("ControlNetLoader", "ControlNetApply",
                                "ControlNetApplyAdvanced"):
                params.setdefault("workflow_type", "controlnet")

        # Detect workflow type if not already set
        if "workflow_type" not in params:
            if has_load_image:
                # Could be img2img, inpaint, or upscale
                if params.get("denoise") is not None and params["denoise"] < 1.0:
                    params["workflow_type"] = "img2img"
                else:
                    params["workflow_type"] = "img2img"
            else:
                params["workflow_type"] = "txt2img"

        return params

    # ================================================================
    # File measurement (no Pillow dependency)
    # ================================================================

    @staticmethod
    def measure_file(filepath: str) -> dict:
        """
        Get file metadata by parsing headers. No Pillow dependency.

        Images: file_size, width, height, media_type='image'
        Videos: file_size, width, height (if MP4), duration (if MP4), media_type='video'
        Audio:  file_size, duration (if WAV), media_type='audio'
        """
        result = {}
        try:
            result["file_size"] = os.path.getsize(filepath)
        except OSError:
            return result

        mt = detect_media_type(filepath)
        result["media_type"] = mt

        try:
            with open(filepath, "rb") as f:
                header = f.read(32)

                # ---- IMAGE FORMATS ----

                # PNG: 8-byte signature + IHDR chunk
                if header[:8] == b"\x89PNG\r\n\x1a\n":
                    if len(header) >= 24:
                        w, h = struct.unpack(">II", header[16:24])
                        result["width"] = w
                        result["height"] = h

                # JPEG: scan for SOF0 marker
                elif header[:2] == b"\xff\xd8":
                    f.seek(0)
                    data = f.read(min(result["file_size"], 65536))
                    i = 2
                    while i < len(data) - 9:
                        if data[i] != 0xFF:
                            i += 1
                            continue
                        marker = data[i + 1]
                        if marker in (0xC0, 0xC1, 0xC2):  # SOF markers
                            h, w = struct.unpack(">HH", data[i + 5:i + 9])
                            result["width"] = w
                            result["height"] = h
                            break
                        if marker == 0xDA:  # SOS — stop scanning
                            break
                        if marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5,
                                      0xD6, 0xD7, 0xD8, 0xD9, 0x01):
                            i += 2
                            continue
                        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
                        i += 2 + seg_len

                # WebP
                elif header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                    if header[12:16] == b"VP8 " and len(header) >= 30:
                        w = struct.unpack("<H", header[26:28])[0] & 0x3FFF
                        h = struct.unpack("<H", header[28:30])[0] & 0x3FFF
                        result["width"] = w
                        result["height"] = h

                # GIF: 6-byte signature + logical screen descriptor
                elif header[:6] in (b"GIF87a", b"GIF89a"):
                    if len(header) >= 10:
                        w, h = struct.unpack("<HH", header[6:10])
                        result["width"] = w
                        result["height"] = h

                # BMP
                elif header[:2] == b"BM":
                    if len(header) >= 26:
                        w, h = struct.unpack("<ii", header[18:26])
                        result["width"] = abs(w)
                        result["height"] = abs(h)

                # ---- VIDEO FORMATS ----

                # MP4/MOV (ftyp box) — parse moov for width/height/duration
                elif header[4:8] in (b"ftyp", b"moov", b"mdat", b"free"):
                    result.update(MediaCatalog._parse_mp4_metadata(f, result["file_size"]))

                # ---- AUDIO FORMATS ----

                # WAV (RIFF...WAVE)
                elif header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                    result.update(MediaCatalog._parse_wav_metadata(f, header))

        except Exception as e:
            logger.debug(f"Could not parse media metadata for {filepath}: {e}")

        return result

    @staticmethod
    def _parse_mp4_metadata(f, file_size: int) -> dict:
        """Extract width, height, duration from MP4/MOV by scanning boxes."""
        result = {}
        try:
            f.seek(0)
            # Scan top-level boxes looking for moov
            pos = 0
            while pos < file_size - 8:
                f.seek(pos)
                box_header = f.read(8)
                if len(box_header) < 8:
                    break
                box_size, box_type = struct.unpack(">I4s", box_header)
                if box_size == 0:
                    break
                if box_size == 1:  # 64-bit extended size
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    box_size = struct.unpack(">Q", ext)[0]

                if box_type == b"moov":
                    # Read moov contents and scan for mvhd (duration) and tkhd (dimensions)
                    moov_data = f.read(min(box_size - 8, 1024 * 1024))  # cap at 1MB
                    # mvhd: duration and timescale
                    mvhd_idx = moov_data.find(b"mvhd")
                    if mvhd_idx >= 4:
                        mvhd_start = mvhd_idx + 4  # past 'mvhd'
                        version = moov_data[mvhd_start] if mvhd_start < len(moov_data) else 0
                        if version == 0 and mvhd_start + 20 <= len(moov_data):
                            timescale = struct.unpack(">I", moov_data[mvhd_start + 12:mvhd_start + 16])[0]
                            duration = struct.unpack(">I", moov_data[mvhd_start + 16:mvhd_start + 20])[0]
                            if timescale > 0:
                                result["duration"] = round(duration / timescale, 2)

                    # tkhd: width and height (first video track)
                    tkhd_idx = moov_data.find(b"tkhd")
                    if tkhd_idx >= 4:
                        tkhd_start = tkhd_idx + 4
                        version = moov_data[tkhd_start] if tkhd_start < len(moov_data) else 0
                        if version == 0 and tkhd_start + 84 <= len(moov_data):
                            # width/height are at offset 76 and 80 as 16.16 fixed-point
                            w_fixed = struct.unpack(">I", moov_data[tkhd_start + 76:tkhd_start + 80])[0]
                            h_fixed = struct.unpack(">I", moov_data[tkhd_start + 80:tkhd_start + 84])[0]
                            w = w_fixed >> 16
                            h = h_fixed >> 16
                            if w > 0 and h > 0:
                                result["width"] = w
                                result["height"] = h
                    break  # found moov, done
                pos += box_size
        except Exception:
            pass
        return result

    @staticmethod
    def _parse_wav_metadata(f, header: bytes) -> dict:
        """Extract duration from WAV file header."""
        result = {}
        try:
            if len(header) >= 28:
                # fmt chunk starts at offset 12 typically
                f.seek(0)
                data = f.read(min(256, os.path.getsize(f.name) if hasattr(f, 'name') else 256))
                fmt_idx = data.find(b"fmt ")
                if fmt_idx >= 0 and fmt_idx + 24 <= len(data):
                    fmt_start = fmt_idx + 8  # past 'fmt ' + chunk size
                    channels = struct.unpack("<H", data[fmt_start + 2:fmt_start + 4])[0]
                    sample_rate = struct.unpack("<I", data[fmt_start + 4:fmt_start + 8])[0]
                    byte_rate = struct.unpack("<I", data[fmt_start + 8:fmt_start + 12])[0]
                    # data chunk for total byte count
                    data_idx = data.find(b"data", fmt_idx)
                    if data_idx >= 0 and data_idx + 8 <= len(data):
                        data_size = struct.unpack("<I", data[data_idx + 4:data_idx + 8])[0]
                        if byte_rate > 0:
                            result["duration"] = round(data_size / byte_rate, 2)
        except Exception:
            pass
        return result

    # ================================================================
    # Retroactive cataloging
    # ================================================================

    def scan_output_directory(self, output_dir: str = None) -> dict:
        """Scan output directory for media files not already cataloged."""
        output_dir = output_dir or self.output_dir
        if not output_dir or not os.path.isdir(output_dir):
            return {"error": "No output directory configured or found", "new_files": 0}

        conn = self._conn()
        try:
            # Get already-known filenames
            known_gen = set()
            for row in conn.execute("SELECT filename FROM generation_files").fetchall():
                known_gen.add(row["filename"])
            known_orphan = set()
            for row in conn.execute("SELECT filename FROM orphan_files").fetchall():
                known_orphan.add(row["filename"])

            new_count = 0
            batch = []

            for root, _dirs, files in os.walk(output_dir):
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in ALL_MEDIA_EXTS:
                        continue
                    if filename in known_gen or filename in known_orphan:
                        continue

                    rel_subfolder = os.path.relpath(root, output_dir)
                    if rel_subfolder == ".":
                        rel_subfolder = ""

                    filepath = os.path.join(root, filename)
                    info = self.measure_file(filepath)
                    fmt = ext.lstrip(".")
                    mt = info.get("media_type", detect_media_type(filename))

                    batch.append((
                        filename, rel_subfolder, filepath,
                        info.get("file_size"), info.get("width"),
                        info.get("height"), fmt, mt, info.get("duration"),
                    ))

            if batch:
                conn.executemany(
                    """INSERT OR IGNORE INTO orphan_files
                       (filename, subfolder, disk_path, file_size, width, height,
                        format, media_type, duration)
                       VALUES (?,?,?,?,?,?, ?,?,?)""",
                    batch,
                )
                new_count = len(batch)
                conn.commit()
            cataloged = len(known_gen)
        finally:
            conn.close()

        # Promote all orphans to generation records so they appear in gallery
        adopted = self.adopt_orphans()

        # Invalidate stats cache after scan
        if new_count > 0 or adopted > 0:
            self._stats_cache = None

        return {
            "new_files": adopted + new_count,
            "total_orphans": 0,
            "already_cataloged": cataloged,
        }

    def adopt_orphans(self) -> int:
        """Promote all orphan files to generation records so they appear in the gallery."""
        conn = self._conn()
        try:
            orphans = conn.execute("SELECT * FROM orphan_files").fetchall()
            if not orphans:
                return 0

            gen_rows = []
            file_rows = []
            orphan_ids = []

            for orphan in orphans:
                orphan = dict(orphan)
                gen_id = str(uuid.uuid4())
                prompt_id = f"scan_{gen_id[:8]}"

                gen_rows.append((
                    gen_id, prompt_id, None, None, "scanned",
                    "completed", orphan.get("filename", ""),
                    orphan.get("width"), orphan.get("height"),
                    orphan.get("discovered_at"),
                ))

                file_rows.append((
                    gen_id, orphan.get("filename", ""),
                    orphan.get("subfolder", ""), "output",
                    orphan.get("disk_path", ""),
                    orphan.get("file_size"), orphan.get("width"),
                    orphan.get("height"), orphan.get("format", "png"),
                    orphan.get("media_type", "image"),
                    orphan.get("duration"), 1,
                ))

                orphan_ids.append((orphan["id"],))

            conn.executemany(
                """INSERT INTO generations
                   (id, prompt_id, instance_id, instance_port, workflow_type,
                    status, prompt_text, width, height, created_at)
                   VALUES (?,?,?,?, ?,?,?,?,?, ?)""",
                gen_rows,
            )
            conn.executemany(
                """INSERT INTO generation_files
                   (generation_id, filename, subfolder, file_type, disk_path,
                    file_size, width, height, format, media_type, duration, is_primary)
                   VALUES (?,?,?,?,?, ?,?,?,?,?,?,?)""",
                file_rows,
            )
            conn.executemany(
                "DELETE FROM orphan_files WHERE id = ?",
                orphan_ids,
            )

            conn.commit()
            adopted = len(orphan_ids)
            if adopted:
                logger.info(f"Adopted {adopted} orphan files into gallery")
            return adopted
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to adopt orphans: {e}")
            return 0
        finally:
            conn.close()

    def get_orphan_files(self, limit: int = 100, offset: int = 0) -> List[dict]:
        conn = self._read()
        rows = conn.execute(
            "SELECT * FROM orphan_files ORDER BY discovered_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    # ================================================================
    # Input pipeline
    # ================================================================

    def copy_to_input(
        self, generation_id: str, file_index: int, comfyui_input_dir: str
    ) -> Optional[str]:
        """Copy a generation output to ComfyUI's input/ folder. Returns new filename."""
        conn = self._read()
        files = conn.execute(
            "SELECT * FROM generation_files WHERE generation_id = ? ORDER BY is_primary DESC, id",
            (generation_id,),
        ).fetchall()
        if not files or file_index >= len(files):
            return None

        src_file = dict(files[file_index])
        src_path = src_file.get("disk_path", "")
        if not src_path or not os.path.exists(src_path):
            return None

        os.makedirs(comfyui_input_dir, exist_ok=True)

        # Prefix with short gen_id to avoid collisions
        short_id = generation_id[:8]
        new_filename = f"{short_id}_{src_file['filename']}"
        dest_path = os.path.join(comfyui_input_dir, new_filename)

        shutil.copy2(src_path, dest_path)
        logger.info(f"Copied {src_path} → {dest_path}")
        return new_filename

    def download_to_input(self, url: str, comfyui_input_dir: str) -> Optional[str]:
        """Download a URL to ComfyUI's input/ folder. Returns filename."""
        os.makedirs(comfyui_input_dir, exist_ok=True)

        # Extract filename from URL or generate one
        url_path = url.split("?")[0].split("/")[-1]
        if "." not in url_path:
            url_path = f"download_{uuid.uuid4().hex[:8]}.png"
        dest_path = os.path.join(comfyui_input_dir, url_path)

        try:
            urlretrieve(url, dest_path)
            logger.info(f"Downloaded {url} → {dest_path}")
            return url_path
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def get_filename_for_generation(
        self, generation_id: str, file_index: int = 0
    ) -> Optional[str]:
        """Resolve a generation_id to a filename."""
        conn = self._read()
        files = conn.execute(
            "SELECT filename FROM generation_files WHERE generation_id = ? "
            "ORDER BY is_primary DESC, id",
            (generation_id,),
        ).fetchall()
        if not files or file_index >= len(files):
            return None
        return files[file_index]["filename"]
