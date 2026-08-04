#!/usr/bin/env python3
"""Validate YOLO-seg label files (class ids + normalized polygon coords)."""

from __future__ import annotations

import argparse
from pathlib import Path


def validate_annotations(label_dir: Path, max_class: int = 4) -> list[str]:
    errors: list[str] = []
    for lbl_path in sorted(label_dir.glob("*.txt")):
        content = lbl_path.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(content.strip().splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            try:
                cls_id = int(parts[0])
            except ValueError:
                errors.append(f"{lbl_path.name}:{i} invalid class")
                continue
            if cls_id < 0 or cls_id > max_class:
                errors.append(f"{lbl_path.name}:{i} class {cls_id} out of range")
            if len(parts) < 7:
                errors.append(f"{lbl_path.name}:{i} need >=3 points")
            for coord in parts[1:]:
                try:
                    val = float(coord)
                except ValueError:
                    errors.append(f"{lbl_path.name}:{i} bad coord {coord}")
                    continue
                if not (0.0 <= val <= 1.0):
                    errors.append(f"{lbl_path.name}:{i} coord {val} out of [0,1]")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--max-class", type=int, default=4)
    args = parser.parse_args()
    errors = validate_annotations(args.labels, args.max_class)
    if errors:
        print(f"Found {len(errors)} issues:")
        for err in errors[:30]:
            print(f"  {err}")
        return 1
    n = len(list(args.labels.glob("*.txt")))
    print(f"OK: {n} label files in {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
