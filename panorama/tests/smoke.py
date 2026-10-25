"""CPU smoke of the preserved core and public CLI, using owned synthetic data."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("panorama_core", ROOT / "CV_work1.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def make_fixture(directory):
    """Three 360x240 crops of one seeded 600x240 procedural scene."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261025)
    scene = rng.integers(30, 226, (240, 600, 3), dtype=np.uint8)
    scene = cv.GaussianBlur(scene, (3, 3), 0.6)
    for i in range(90):
        x, y = rng.integers([5, 5], [595, 235])
        color = tuple(int(c) for c in rng.integers(10, 245, 3))
        cv.circle(scene, (int(x), int(y)), int(rng.integers(3, 12)), color, 2)
    for i, x in enumerate(range(15, 580, 70)):
        cv.putText(scene, str(i), (x, 125), cv.FONT_HERSHEY_SIMPLEX, 0.8, (250, 15, 240), 2)
    paths = []
    for i, x in enumerate([0, 120, 240], 1):
        path = directory / f"{i}.jpg"
        assert cv.imwrite(str(path), scene[:, x:x + 360], [cv.IMWRITE_JPEG_QUALITY, 98])
        paths.append(path)
    return paths


def run(write_demo=None):
    cv.setNumThreads(1)
    cv.setRNGSeed(20261025)
    with tempfile.TemporaryDirectory(prefix="panorama_smoke_") as temporary:
        work = Path(temporary)
        inputs = make_fixture(work / "inputs")
        images = [cv.imread(str(p)) for p in inputs]
        # Known geometry checks real SIFT / FLANN / RANSAC, not mocked calls.
        pair_metrics = []
        points = np.float32([[10, 10], [350, 10], [350, 230], [10, 230], [180, 120]]).reshape(-1, 1, 2)
        for i, dx in [(0, -120), (2, 120)]:
            H, nf, ni, probability = core.find_transform(images[i], images[1])
            assert H is not None and np.isfinite(H).all(), "missing/invalid homography"
            assert nf >= 20 and ni >= 20, "too few synthetic matches"
            expected = points + np.array([dx, 0], dtype=np.float32)
            error = float(np.linalg.norm(cv.perspectiveTransform(points, H) - expected, axis=2).max())
            assert error < 2.0, f"known geometry error: {error} pixels"
            pair_metrics.append({"pair": "A-B" if i == 0 else "C-B", "nf": nf, "ni": ni,
                                 "model_score": probability, "max_control_point_error_px": error})
        crop, full, used = core.stitch_three_images(images)
        assert full.dtype == np.uint8 and crop.dtype == np.uint8
        assert 595 <= full.shape[1] <= 605 and 235 <= full.shape[0] <= 245
        assert crop.shape[1] >= 580 and crop.shape[0] >= 230
        assert float(np.std(crop)) > 10 and len(used) == 2
        reference_shapes = {}
        for filename, function in [("CV_work1_rest.py", "stitch_to_middle"),
                                    ("CV_work1_0.py", "image_stitching"),
                                    ("Pasted_code_commented.py", "stitch_three_images")]:
            reference_spec = importlib.util.spec_from_file_location("reference_core", ROOT / "reference" / filename)
            reference = importlib.util.module_from_spec(reference_spec)
            reference_spec.loader.exec_module(reference)
            cv.setRNGSeed(20261025)
            output = getattr(reference, function)(images)
            image = output if isinstance(output, np.ndarray) else output[0]
            assert image.dtype == np.uint8 and image.shape[1] >= 580 and image.shape[0] >= 230
            reference_shapes[filename] = list(image.shape)
        assert core.find_transform(np.zeros((32, 32, 3), np.uint8), images[1])[0] is None
        try:
            core.stitch_three_images([np.zeros((32, 32, 3), np.uint8)] * 3)
        except ValueError:
            pass
        else:
            raise AssertionError("textureless input must fail instead of producing a panorama")
        # Public entry point must read supplied files and save both actual core outputs.
        result = subprocess.run([sys.executable, str(ROOT / "run_panorama.py"),
                                 *(str(p) for p in inputs), "--output-dir", str(work / "output"),
                                 "--max-side", "320"], capture_output=True, text=True,
                                encoding="utf-8", timeout=60)
        assert result.returncode == 0, f"public entry failed: {result.stderr}"
        cli_shapes = {}
        for name in ["result_crop.jpg", "result_full.jpg"]:
            image = cv.imread(str(work / "output" / name))
            assert image is not None and image.shape[1] >= 500 and image.shape[0] >= 195
            cli_shapes[name] = list(image.shape)
        invalid = subprocess.run([sys.executable, str(ROOT / "run_panorama.py"),
                                  str(work / "missing.jpg"), str(inputs[1]), str(inputs[2]),
                                  "--output-dir", str(work / "invalid")], capture_output=True,
                                 text=True, encoding="utf-8", timeout=30)
        assert invalid.returncode != 0 and not (work / "invalid" / "result_full.jpg").exists()
        if write_demo:
            demo_inputs = make_fixture(write_demo)
            subprocess.run([sys.executable, str(ROOT / "run_panorama.py"),
                            *(str(p) for p in demo_inputs), "--output-dir", str(write_demo)],
                           check=True, stdout=subprocess.DEVNULL, timeout=60)
        return {"status": "PASS", "scope": "synthetic CPU smoke, not a performance benchmark",
                "python": sys.version.split()[0], "opencv": cv.__version__, "numpy": np.__version__,
                "seed": 20261025, "input_shape_hwc": list(images[0].shape),
                "core_full_shape_hwc": list(full.shape), "core_crop_shape_hwc": list(crop.shape),
                "pairs": pair_metrics, "cli_shapes_hwc": cli_shapes,
                "reference_shapes_hwc": reference_shapes,
                "checks": ["known homographies within 2 px", "real warp/feather/crop",
                           "blank-image rejection", "CLI saved both outputs with resize", "missing-input rejection"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-demo", type=Path, help="also regenerate the owned demo images and outputs")
    args = parser.parse_args()
    print(json.dumps(run(args.write_demo.resolve() if args.write_demo else None), indent=2))
