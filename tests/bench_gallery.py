"""
Benchmark gallery performance with 10k generation records.

Tests:
1. Database query speed (pagination, search, filters)
2. Image proxy throughput (concurrent file requests)
3. Stats endpoint under load
"""
import asyncio
import json
import os
import random
import shutil
import sqlite3
import string
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Checkpoints to simulate
CHECKPOINTS = [
    "dreamshaper_8.safetensors",
    "flux-dev-Q5_K_M.gguf",
    "sd_xl_base_1.0.safetensors",
    "realisticVision_v51.safetensors",
    "deliberate_v3.safetensors",
    "anythingV5.safetensors",
    "revAnimated_v122.safetensors",
    "photon_v1.safetensors",
]

TAGS_POOL = [
    "landscape", "portrait", "scifi", "fantasy", "photo", "anime",
    "abstract", "nature", "urban", "concept-art", "character", "vehicle",
]

SAMPLERS = ["euler", "euler_a", "dpm_2", "dpmpp_2m", "dpmpp_sde", "uni_pc"]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform"]


def seed_database(db_path: str, n: int = 10000):
    """Insert N generation records + files into the database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")  # Safe for seeding
    cursor = conn.cursor()

    # Create tables
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
            media_type      TEXT DEFAULT 'image',
            format          TEXT DEFAULT 'png',
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

    # Prepare batch data
    gen_rows = []
    file_rows = []
    base_time = time.time() - (n * 60)  # Spread over N minutes

    prompts = [
        "a beautiful {adj} {subject} in {style} style, {quality}",
        "photo of {subject} with {adj} lighting, {style}",
        "{adj} {subject}, digital art, {style}, {quality}",
        "cinematic {subject} scene, {adj} atmosphere, {style}",
    ]
    adjs = ["stunning", "ethereal", "dark", "vibrant", "moody", "dreamy", "epic", "serene"]
    subjects = ["landscape", "portrait", "city", "forest", "ocean", "mountain", "castle", "robot"]
    styles = ["photorealistic", "anime", "oil painting", "watercolor", "concept art", "pixel art"]
    qualities = ["masterpiece", "best quality", "8k", "highly detailed", "professional"]

    for i in range(n):
        gen_id = f"gen-{uuid.uuid4().hex[:12]}"
        prompt_id = f"pid-{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(base_time + i * 60))

        prompt_text = random.choice(prompts).format(
            adj=random.choice(adjs),
            subject=random.choice(subjects),
            style=random.choice(styles),
            quality=random.choice(qualities),
        )

        # Some generations are favorites, some have ratings
        is_fav = 1 if random.random() < 0.08 else 0
        rating = random.choices([0, 1, 2, 3, 4, 5], weights=[60, 10, 10, 10, 5, 5])[0]
        tags = ",".join(random.sample(TAGS_POOL, random.randint(1, 4)))
        checkpoint = random.choice(CHECKPOINTS)
        sampler = random.choice(SAMPLERS)
        scheduler = random.choice(SCHEDULERS)
        width = random.choice([512, 768, 1024])
        height = random.choice([512, 768, 1024])

        gen_rows.append((
            gen_id, prompt_id, "inst-1", 8188, None, "txt2img", "completed",
            checkpoint, prompt_text, "blurry, low quality",
            random.randint(1, 999999999), random.choice([20, 25, 30, 40]),
            random.choice([5.0, 7.0, 7.5, 8.0, 12.0]),
            sampler, scheduler, None, width, height, None,
            None, tags, rating, is_fav, None, ts, ts, None,
        ))

        # 1-3 files per generation
        num_files = random.choices([1, 2, 3], weights=[70, 20, 10])[0]
        for fi in range(num_files):
            filename = f"ComfyUI_{i:05d}_{fi:02d}.png"
            file_rows.append((
                gen_id, filename, "", "output", None,
                random.randint(100000, 2000000),  # 100KB-2MB
                width, height, "png", "image", None,
                1 if fi == 0 else 0, ts,
            ))

    # Batch insert
    cursor.executemany(
        """INSERT INTO generations
        (id, prompt_id, instance_id, instance_port, parent_id, workflow_type, status,
         checkpoint, prompt_text, negative_prompt, seed, steps, cfg, sampler, scheduler,
         denoise, width, height, workflow_json, title, tags, rating, favorite, notes,
         created_at, updated_at, comfyui_dir)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        gen_rows,
    )
    cursor.executemany(
        """INSERT INTO generation_files
        (generation_id, filename, subfolder, file_type, disk_path, file_size,
         width, height, format, media_type, duration, is_primary, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        file_rows,
    )

    conn.commit()
    conn.close()
    return len(gen_rows), len(file_rows)


def bench_direct_sql(db_path: str):
    """Benchmark raw SQL query performance."""
    print("\n1. Direct SQL benchmark (baseline):")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Page 1 (newest)
    start = time.perf_counter()
    for _ in range(20):
        rows = conn.execute(
            """SELECT g.*, gf.filename as primary_filename,
                gf.subfolder as primary_subfolder,
                gf.media_type as primary_media_type,
                gf.duration as primary_duration
                FROM generations g
                LEFT JOIN generation_files gf ON gf.generation_id = g.id AND gf.is_primary = 1
                WHERE status != 'failed'
                ORDER BY created_at DESC
                LIMIT 24 OFFSET 0"""
        ).fetchall()
    elapsed = time.perf_counter() - start
    print(f"  Page 1 (newest, 20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/query)")

    # Page 200 (deep pagination)
    start = time.perf_counter()
    for _ in range(20):
        rows = conn.execute(
            """SELECT g.*, gf.filename as primary_filename,
                gf.subfolder as primary_subfolder,
                gf.media_type as primary_media_type,
                gf.duration as primary_duration
                FROM generations g
                LEFT JOIN generation_files gf ON gf.generation_id = g.id AND gf.is_primary = 1
                WHERE status != 'failed'
                ORDER BY created_at DESC
                LIMIT 24 OFFSET 4800"""
        ).fetchall()
    elapsed = time.perf_counter() - start
    print(f"  Page 200 (deep, 20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/query)")

    # COUNT query
    start = time.perf_counter()
    for _ in range(20):
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM generations WHERE status != 'failed'"
        ).fetchone()["cnt"]
    elapsed = time.perf_counter() - start
    print(f"  COUNT(*) (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/query, total={total})")

    # Search with LIKE
    start = time.perf_counter()
    for _ in range(20):
        rows = conn.execute(
            """SELECT g.*, gf.filename as primary_filename,
                gf.subfolder as primary_subfolder,
                gf.media_type as primary_media_type,
                gf.duration as primary_duration
                FROM generations g
                LEFT JOIN generation_files gf ON gf.generation_id = g.id AND gf.is_primary = 1
                WHERE status != 'failed' AND prompt_text LIKE ?
                ORDER BY created_at DESC
                LIMIT 24 OFFSET 0""",
            ["%landscape%"],
        ).fetchall()
    elapsed = time.perf_counter() - start
    print(f"  LIKE search (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/query, {len(rows)} results)")

    # Checkpoint filter
    start = time.perf_counter()
    for _ in range(20):
        rows = conn.execute(
            """SELECT g.*, gf.filename as primary_filename,
                gf.subfolder as primary_subfolder,
                gf.media_type as primary_media_type,
                gf.duration as primary_duration
                FROM generations g
                LEFT JOIN generation_files gf ON gf.generation_id = g.id AND gf.is_primary = 1
                WHERE status != 'failed' AND checkpoint LIKE ?
                ORDER BY created_at DESC
                LIMIT 24 OFFSET 0""",
            ["%flux%"],
        ).fetchall()
    elapsed = time.perf_counter() - start
    print(f"  Checkpoint filter (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/query)")

    # Combined COUNT + search (what search_generations does)
    start = time.perf_counter()
    for _ in range(20):
        rows = conn.execute(
            """SELECT g.*, gf.filename as primary_filename,
                gf.subfolder as primary_subfolder,
                gf.media_type as primary_media_type,
                gf.duration as primary_duration
                FROM generations g
                LEFT JOIN generation_files gf ON gf.generation_id = g.id AND gf.is_primary = 1
                WHERE status != 'failed'
                ORDER BY created_at DESC
                LIMIT 24 OFFSET 0"""
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM generations WHERE status != 'failed'"
        ).fetchone()["cnt"]
    elapsed = time.perf_counter() - start
    print(f"  Page+COUNT combined (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/query)")

    conn.close()


def bench_media_catalog(db_path: str, output_dir: str):
    """Benchmark via MediaCatalog class."""
    from backend.media_catalog import MediaCatalog

    print("\n2. MediaCatalog.search_generations() benchmark:")
    catalog = MediaCatalog(db_path, output_dir)

    # Page 1
    start = time.perf_counter()
    for _ in range(20):
        results, total = catalog.search_generations(limit=24, offset=0)
    elapsed = time.perf_counter() - start
    print(f"  Page 1 (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/call, {total} total)")

    # Page 200
    start = time.perf_counter()
    for _ in range(20):
        results, total = catalog.search_generations(limit=24, offset=4800)
    elapsed = time.perf_counter() - start
    print(f"  Page 200 (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/call)")

    # Search
    start = time.perf_counter()
    for _ in range(20):
        results, total = catalog.search_generations(query="landscape", limit=24)
    elapsed = time.perf_counter() - start
    print(f"  Text search (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/call, {total} matches)")

    # Checkpoint + tags filter
    start = time.perf_counter()
    for _ in range(20):
        results, total = catalog.search_generations(
            checkpoint="flux", tags="scifi", limit=24
        )
    elapsed = time.perf_counter() - start
    print(f"  Checkpoint+tags (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/call, {total} matches)")

    # Favorites only
    start = time.perf_counter()
    for _ in range(20):
        results, total = catalog.search_generations(favorite=True, limit=24)
    elapsed = time.perf_counter() - start
    print(f"  Favorites only (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/call, {total} matches)")

    # Stats
    start = time.perf_counter()
    for _ in range(20):
        catalog._stats_cache = None  # Bypass cache
        stats = catalog.get_stats()
    elapsed = time.perf_counter() - start
    print(f"  get_stats() uncached (20x): {elapsed*1000:.1f}ms ({elapsed/20*1000:.2f}ms/call)")

    start = time.perf_counter()
    for _ in range(200):
        stats = catalog.get_stats()
    elapsed = time.perf_counter() - start
    print(f"  get_stats() cached (200x): {elapsed*1000:.1f}ms ({elapsed/200*1000:.3f}ms/call)")


async def bench_http_gallery(n_concurrent: int = 20, n_requests: int = 200):
    """Benchmark gallery endpoint via HTTP with concurrent requests."""
    import httpx

    print(f"\n3. HTTP gallery endpoint ({n_concurrent} concurrent, {n_requests} total):")

    async with httpx.AsyncClient(
        base_url="http://localhost:8000",
        timeout=30.0,
        limits=httpx.Limits(max_connections=n_concurrent, max_keepalive_connections=n_concurrent),
    ) as client:
        # Test: gallery page 1
        latencies = []
        sem = asyncio.Semaphore(n_concurrent)

        async def fetch_page(page: int):
            async with sem:
                start = time.perf_counter()
                resp = await client.get(
                    "/api/comfyui/media/generations",
                    params={"limit": 24, "offset": page * 24, "sort": "newest"},
                )
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                return resp.status_code

        # Random pages
        pages = [random.randint(0, 400) for _ in range(n_requests)]
        start = time.perf_counter()
        results = await asyncio.gather(*[fetch_page(p) for p in pages])
        total_elapsed = time.perf_counter() - start

        ok = sum(1 for r in results if r == 200)
        errors = sum(1 for r in results if r != 200)
        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000
        avg = sum(latencies) / len(latencies) * 1000
        rps = n_requests / total_elapsed

        print(f"  Random pages: {ok}/{n_requests} OK, {errors} errors")
        print(f"  Latency: avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms")
        print(f"  Throughput: {rps:.0f} req/s ({total_elapsed:.1f}s total)")

        # Test: search queries
        latencies = []
        queries = ["landscape", "portrait", "ocean", "robot", "anime", "dark"]

        async def fetch_search(q: str):
            async with sem:
                start = time.perf_counter()
                resp = await client.get(
                    "/api/comfyui/media/generations",
                    params={"query": q, "limit": 24, "offset": 0},
                )
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                return resp.status_code

        search_tasks = [fetch_search(random.choice(queries)) for _ in range(100)]
        start = time.perf_counter()
        results = await asyncio.gather(*search_tasks)
        total_elapsed = time.perf_counter() - start

        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000
        avg = sum(latencies) / len(latencies) * 1000

        print(f"  Search queries: avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms")

        # Test: stats endpoint burst
        latencies = []

        async def fetch_stats():
            async with sem:
                start = time.perf_counter()
                resp = await client.get("/api/comfyui/media/stats")
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                return resp.status_code

        start = time.perf_counter()
        results = await asyncio.gather(*[fetch_stats() for _ in range(200)])
        total_elapsed = time.perf_counter() - start

        latencies.sort()
        avg = sum(latencies) / len(latencies) * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000

        print(f"  Stats burst (200x): avg={avg:.1f}ms  p99={p99:.1f}ms  {200/total_elapsed:.0f} req/s")


async def bench_image_proxy(output_dir: str, n_concurrent: int = 20, n_requests: int = 200):
    """Benchmark image proxy throughput."""
    import httpx

    # Get list of files
    files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    if not files:
        print("\n4. Image proxy: No files to test (skipping)")
        return

    print(f"\n4. Image proxy ({n_concurrent} concurrent, {n_requests} total, {len(files)} files):")

    async with httpx.AsyncClient(
        base_url="http://localhost:8000",
        timeout=30.0,
        limits=httpx.Limits(max_connections=n_concurrent, max_keepalive_connections=n_concurrent),
    ) as client:
        latencies = []
        bytes_total = 0
        sem = asyncio.Semaphore(n_concurrent)

        async def fetch_image(filename: str):
            nonlocal bytes_total
            async with sem:
                start = time.perf_counter()
                resp = await client.get(f"/api/comfyui/images/{filename}")
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                if resp.status_code == 200:
                    bytes_total += len(resp.content)
                return resp.status_code

        tasks = [fetch_image(random.choice(files)) for _ in range(n_requests)]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start

        ok = sum(1 for r in results if r == 200)
        errors = sum(1 for r in results if r != 200)
        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000
        avg = sum(latencies) / len(latencies) * 1000
        rps = n_requests / total_elapsed

        print(f"  Results: {ok}/{n_requests} OK, {errors} errors")
        print(f"  Latency: avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms")
        print(f"  Throughput: {rps:.0f} req/s  {bytes_total/1024/1024:.1f}MB transferred")


def create_dummy_images(output_dir: str, n: int = 100):
    """Create minimal valid PNG files for image proxy testing."""
    os.makedirs(output_dir, exist_ok=True)
    # Minimal 1x1 red PNG (67 bytes)
    import struct, zlib
    def make_png(width=64, height=64):
        def chunk(chunk_type, data):
            c = chunk_type + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
        raw = b''
        for _ in range(height):
            raw += b'\x00' + bytes([random.randint(0, 255) for _ in range(width * 3)])
        idat = zlib.compress(raw)
        return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')

    png_data = make_png()
    for i in range(n):
        filepath = os.path.join(output_dir, f"ComfyUI_{i:05d}_00.png")
        with open(filepath, 'wb') as f:
            f.write(png_data)
    return n


async def main():
    print("=" * 65)
    print("  Gallery Performance Benchmark (10k records)")
    print("=" * 65)

    # Setup temp dir
    tmpdir = tempfile.mkdtemp(prefix="gallery_bench_")
    db_path = os.path.join(tmpdir, "media_catalog.db")
    output_dir = os.path.join(tmpdir, "output")

    try:
        # Seed database
        print("\nSeeding database with 10,000 generations...")
        start = time.perf_counter()
        n_gens, n_files = seed_database(db_path, 10000)
        elapsed = time.perf_counter() - start
        print(f"  Seeded {n_gens} generations + {n_files} files in {elapsed:.1f}s")

        db_size = os.path.getsize(db_path)
        print(f"  Database size: {db_size / 1024 / 1024:.1f}MB")

        # Test 1: Raw SQL
        bench_direct_sql(db_path)

        # Test 2: MediaCatalog class
        bench_media_catalog(db_path, output_dir)

        # Create dummy images for proxy test
        print("\nCreating 100 dummy PNG files...")
        n_imgs = create_dummy_images(output_dir, 100)
        print(f"  Created {n_imgs} files")

        # Test 3 & 4: HTTP tests (requires running server)
        # Check if server is running
        try:
            import httpx
            resp = httpx.get("http://localhost:8000/api/models/loaded", timeout=2)
            server_running = resp.status_code == 200
        except Exception:
            server_running = False

        if server_running:
            print("\n  (Server detected — but using temp DB, skipping HTTP tests)")
            print("  Run HTTP tests separately with production DB")
        else:
            print("\n  (Server not running — skipping HTTP tests)")

        print("\n" + "=" * 65)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
