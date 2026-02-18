"""
Benchmark media catalog scan with many files.
Creates temp PNG files and times the scan_output_directory operation.
"""
import os
import sys
import time
import struct
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.media_catalog import MediaCatalog

# Create minimal valid PNG files (8 + 25 = 33 bytes)
def make_png(path, w=512, h=512):
    """Write a minimal PNG header (just enough for measure_file to parse)."""
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk (width + height only needed for parsing)
    ihdr_data = struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00"
    import zlib
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xffffffff)
    ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc
    with open(path, "wb") as f:
        f.write(sig + ihdr)


def main():
    FILE_COUNTS = [100, 1000, 5000]

    for count in FILE_COUNTS:
        # Setup: create temp dir with PNGs
        tmpdir = tempfile.mkdtemp(prefix="media_bench_")
        db_path = os.path.join(tmpdir, "bench.db")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        print(f"\n--- {count} files ---")

        # Create fake PNG files
        t0 = time.perf_counter()
        for i in range(count):
            make_png(os.path.join(output_dir, f"cat_{i:05d}.png"))
        t_create = time.perf_counter() - t0
        print(f"  File creation: {t_create*1000:.0f}ms ({count/t_create:.0f} files/s)")

        # Scan
        catalog = MediaCatalog(db_path, output_dir)
        t0 = time.perf_counter()
        result = catalog.scan_output_directory(output_dir)
        t_scan = time.perf_counter() - t0
        rate = count / t_scan if t_scan > 0 else 0
        print(f"  Scan: {t_scan*1000:.0f}ms ({rate:.0f} files/s)")
        print(f"  Result: {result}")

        # Second scan (all already known)
        t0 = time.perf_counter()
        result2 = catalog.scan_output_directory(output_dir)
        t_rescan = time.perf_counter() - t0
        print(f"  Re-scan (no new): {t_rescan*1000:.0f}ms")

        # Cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
