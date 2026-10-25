"""Path-only public entry adapter; the archived algorithm remains unchanged."""
import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import CV_work1 as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs=3, metavar="IMAGE",
                        help="left, middle and right images in that order")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="save result_crop.jpg and result_full.jpg here")
    parser.add_argument("--max-side", type=int, default=1600,
                        help="resize each input's longest side (default: 1600)")
    args = parser.parse_args()
    if args.max_side <= 0:
        parser.error("--max-side must be positive")
    paths = [p.resolve() for p in args.images]
    for path in paths:
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    output = args.output_dir.resolve()
    targets = [output / name for name in ("result_crop.jpg", "result_full.jpg")]
    if any(target in paths for target in targets):
        parser.error("output filenames must not overwrite input images")
    output.mkdir(parents=True, exist_ok=True)
    core.IMAGE_PATHS = [str(p) for p in paths]
    core.MAX_SIDE = args.max_side
    previous = Path.cwd()
    try:
        os.chdir(output)
        core.main()
    finally:
        os.chdir(previous)
    # The archived core does not inspect cv.imwrite's return value.
    for target in targets:
        if not target.is_file():
            raise RuntimeError(f"missing output: {target}")


if __name__ == "__main__":
    main()
