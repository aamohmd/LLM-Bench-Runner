#!/usr/bin/env python3
"""Ship the same deterministic scoring package in a platform checkout."""

import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--platform", type=Path, required=True, help="Path to the platform checkout")
parser.add_argument("--check", action="store_true", help="Verify package equality without writing")
args = parser.parse_args()
source = Path(__file__).resolve().parents[1] / "benchmark_scoring"
platform = args.platform.resolve()
if not (platform / "backend/app/main.py").is_file():
    parser.error("--platform must identify a platform checkout")
destination = platform / "benchmark_scoring"
files = sorted(source.glob("*.py"))
if args.check:
    if not destination.is_dir() or {p.name for p in files} != {p.name for p in destination.glob("*.py")} or any(not (destination / p.name).is_file() or p.read_bytes() != (destination / p.name).read_bytes() for p in files):
        parser.exit(1, "Scoring packages differ; sync before releasing.\n")
    print("Scoring packages match.")
else:
    destination.mkdir(exist_ok=True)
    for target in destination.glob("*.py"):
        if target.name not in {p.name for p in files}:
            target.unlink()
    for file in files:
        shutil.copy2(file, destination / file.name)
    print("Scoring package synchronized; rebuild the platform before use.")
