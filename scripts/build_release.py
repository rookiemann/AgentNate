"""
AgentNate Release Builder
Creates release archives for GitHub Releases.

Output: releases/
  AgentNate-python-core.7z     - Python 3.14.2 + pip + all packages (minus PyTorch)
  AgentNate-torch-cuda.7z      - PyTorch CUDA (separate, ~4.2 GB raw → ~1.5 GB compressed)
  AgentNate-playwright.7z      - Playwright Chromium browser
  AgentNate-node-n8n.7z        - Node.js 24 + n8n (node/ + node_modules/)
  AgentNate-vllm-env.7z        - vLLM Python environment (optional)
  RELEASE_NOTES.md             - Template release notes

GitHub Release asset limit: 2 GB per file.

Usage:
  python scripts/build_release.py                  # Build all archives
  python scripts/build_release.py --only python    # Build only python archives
  python scripts/build_release.py --only node      # Build only node/n8n archive
  python scripts/build_release.py --only vllm      # Build only vLLM archive
  python scripts/build_release.py --skip-vllm      # Build all except vLLM
  python scripts/build_release.py --dry-run        # Show what would be built
"""
import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RELEASES_DIR = ROOT / "releases"

# 7-Zip path
SEVEN_ZIP = r"C:\Program Files\7-Zip\7z.exe"

# Archive definitions: (name, source_paths, exclude_patterns)
ARCHIVES = {
    "python-core": {
        "filename": "AgentNate-python-core.7z",
        "description": "Python 3.14.2 embedded + pip + all packages (excluding PyTorch and Playwright browsers)",
        "sources": ["python/"],
        "excludes": [
            "python/Lib/site-packages/torch*",
            "python/Lib/site-packages/nvidia*",
            "python/Lib/site-packages/triton*",
            "python/.playwright-browsers/",
            "python/Doc/",
        ],
        "extract_to": "Extract into AgentNate/ root (creates python/ folder)",
    },
    "torch-cuda": {
        "filename": "AgentNate-torch-cuda.7z",
        "description": "PyTorch CUDA packages (torch, nvidia, triton)",
        "sources": [
            "python/Lib/site-packages/torch/",
            "python/Lib/site-packages/torch*.dist-info/",
            "python/Lib/site-packages/nvidia/",
            "python/Lib/site-packages/nvidia*.dist-info/",
            "python/Lib/site-packages/triton/",
            "python/Lib/site-packages/triton*.dist-info/",
        ],
        "excludes": [],
        "extract_to": "Extract into AgentNate/ root (merges into python/Lib/site-packages/)",
    },
    "playwright": {
        "filename": "AgentNate-playwright.7z",
        "description": "Playwright Chromium browser for browser automation tools",
        "sources": ["python/.playwright-browsers/"],
        "excludes": [],
        "extract_to": "Extract into AgentNate/ root (creates python/.playwright-browsers/)",
    },
    "node-n8n": {
        "filename": "AgentNate-node-n8n.7z",
        "description": "Node.js 24.12.0 portable + n8n workflow engine",
        "sources": ["node/", "node_modules/"],
        "excludes": [],
        "extract_to": "Extract into AgentNate/ root (creates node/ and node_modules/)",
    },
    "vllm-env": {
        "filename": "AgentNate-vllm-env.7z",
        "description": "vLLM Python 3.10.6 environment for high-throughput serving (optional)",
        "sources": ["envs/vllm/"],
        "excludes": [],
        "extract_to": "Extract into AgentNate/ root (creates envs/vllm/)",
    },
}


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    if not path.exists():
        return 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024**2):.0f} MB"
    else:
        return f"{size_bytes / 1024:.0f} KB"


def check_7zip():
    """Verify 7-Zip is available."""
    if not os.path.exists(SEVEN_ZIP):
        print(f"Error: 7-Zip not found at {SEVEN_ZIP}")
        print("Install 7-Zip from https://www.7-zip.org/ or update SEVEN_ZIP path in this script")
        sys.exit(1)


def build_archive(name: str, config: dict, dry_run: bool = False):
    """Build a single release archive."""
    filename = config["filename"]
    output_path = RELEASES_DIR / filename

    # Check source paths exist
    existing_sources = []
    for src in config["sources"]:
        src_path = ROOT / src.rstrip("/").split("*")[0]  # Handle glob patterns
        if src_path.exists():
            existing_sources.append(src)
        else:
            print(f"  Warning: {src} not found, skipping")

    if not existing_sources:
        print(f"  SKIP: No source paths found for {name}")
        return False

    # Calculate raw size
    raw_size = 0
    for src in existing_sources:
        src_path = ROOT / src.rstrip("/").rstrip("*")
        if src_path.is_dir():
            raw_size += get_dir_size(src_path)
        elif src_path.is_file():
            raw_size += src_path.stat().st_size

    print(f"\n{'='*60}")
    print(f"  {name}: {config['description']}")
    print(f"  Raw size: {format_size(raw_size)}")
    print(f"  Output: {filename}")

    if dry_run:
        print(f"  [DRY RUN] Would create {filename}")
        return True

    # Remove existing archive
    if output_path.exists():
        output_path.unlink()

    # Build 7z command
    cmd = [SEVEN_ZIP, "a", "-t7z", "-mx=5", "-mmt=on", str(output_path)]

    # Add source paths (relative to ROOT)
    for src in existing_sources:
        cmd.append(src)

    # Add excludes
    for excl in config.get("excludes", []):
        cmd.append(f"-xr!{excl}")

    print(f"  Compressing...")
    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:500]}")
        return False

    elapsed = time.time() - start
    compressed_size = output_path.stat().st_size
    ratio = (1 - compressed_size / raw_size) * 100 if raw_size > 0 else 0

    print(f"  Done: {format_size(compressed_size)} ({ratio:.0f}% compression) in {elapsed:.0f}s")

    if compressed_size > 2 * 1024 ** 3:
        print(f"  WARNING: {format_size(compressed_size)} exceeds GitHub's 2 GB limit!")
        print(f"  Consider splitting or using higher compression (-mx=9)")

    return True


def generate_release_notes(version: str = "2.0.0"):
    """Generate template release notes."""
    notes = f"""# AgentNate v{version}

## What's New

AgentNate is a **local-first AI orchestration platform** — LLMs, workflow automation, image/video generation, TTS, and music generation in one portable folder.

### Highlights
- 187+ agent tools across 19 categories
- Multi-provider LLM support (llama.cpp, LM Studio, Ollama, vLLM, OpenRouter)
- Full n8n workflow integration (72 node types, natural language → deployed workflows)
- ComfyUI creative engine (images, video, upscaling, inpainting, ControlNet)
- Text-to-Speech (XTTS v2, Fish Speech, Kokoro) and Music Generation (Stable Audio, ACE-Step)
- Multi-panel concurrent chat with per-panel model/persona selection
- Tool-level race execution for creative tasks
- GGUF model search and download from HuggingFace
- OpenAI-compatible API (`/v1/chat/completions`)
- 100% portable — no system installs, no Docker, no admin rights

## Installation

### Option A: Automatic (Recommended)
```
git clone https://github.com/rookiemann/AgentNate.git
cd AgentNate
install.bat
launcher.bat
```

### Option B: Pre-Built Environments
If `install.bat` has trouble downloading or building dependencies, download the pre-built archives below and extract them into the AgentNate folder:

| Archive | Contents | Required? |
|---------|----------|-----------|
| `AgentNate-python-core.7z` | Python 3.14.2 + pip + all packages | Yes |
| `AgentNate-torch-cuda.7z` | PyTorch CUDA (GPU acceleration) | Yes for GPU |
| `AgentNate-playwright.7z` | Chromium browser (browser automation) | Optional |
| `AgentNate-node-n8n.7z` | Node.js 24 + n8n workflow engine | Yes |
| `AgentNate-vllm-env.7z` | vLLM high-throughput serving | Optional |

**Extract each archive into the AgentNate root folder** so that `python/`, `node/`, and `node_modules/` appear at the top level. Then run `launcher.bat`.

## Requirements
- Windows 10 or 11 (x64)
- NVIDIA GPU recommended (8GB+ VRAM for local models)
- ~2 GB disk space for source + pre-built environments

## Documentation
- [User Manual (Wiki)](https://github.com/rookiemann/AgentNate/wiki)
- [PDF Manual](manual/AgentNate-Manual.pdf) (included in source)
"""

    notes_path = RELEASES_DIR / "RELEASE_NOTES.md"
    notes_path.write_text(notes, encoding="utf-8")
    print(f"\nRelease notes: {notes_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AgentNate Release Builder")
    parser.add_argument("--only", choices=["python", "node", "vllm", "all"], default="all",
                        help="Build only specific archives")
    parser.add_argument("--skip-vllm", action="store_true", help="Skip vLLM environment")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be built")
    parser.add_argument("--version", default="2.0.0", help="Version string for release notes")
    args = parser.parse_args()

    check_7zip()

    # Create output directory
    RELEASES_DIR.mkdir(exist_ok=True)

    print("AgentNate Release Builder")
    print(f"Output: {RELEASES_DIR}/")

    # Determine which archives to build
    to_build = []
    if args.only == "python" or args.only == "all":
        to_build.extend(["python-core", "torch-cuda", "playwright"])
    if args.only == "node" or args.only == "all":
        to_build.append("node-n8n")
    if (args.only == "vllm" or args.only == "all") and not args.skip_vllm:
        to_build.append("vllm-env")

    built = 0
    for name in to_build:
        config = ARCHIVES[name]
        if build_archive(name, config, dry_run=args.dry_run):
            built += 1

    # Generate release notes
    generate_release_notes(args.version)

    # Summary
    print(f"\n{'='*60}")
    print(f"Release build complete: {built}/{len(to_build)} archives")
    if not args.dry_run:
        total_size = sum(
            f.stat().st_size for f in RELEASES_DIR.glob("*.7z") if f.is_file()
        )
        print(f"Total release size: {format_size(total_size)}")
    print(f"\nTo upload: gh release create v{args.version} releases/*.7z releases/RELEASE_NOTES.md")


if __name__ == "__main__":
    main()
