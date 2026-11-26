"""Train a small fish detector from the labeled images.

    python train.py --prepare-only   # just build the train/validation split
    python train.py                  # build the split, train, and check it
"""

import argparse
import json
import os
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATASET_DIR = DATA_DIR / "dataset"  # rebuilt on every run
MODEL_DIR = DATA_DIR / "models"
RUNS_DIR = DATA_DIR / "runs"


def split_images(images, val_fraction):
    """Split by time, never at random: neighboring frames look almost the same.

    With several videos, whole videos are held out. With only one, the last
    part of it is held out.
    """
    videos = {}
    for image in sorted(images):
        videos.setdefault(image.stem.rsplit("_", 1)[0], []).append(image)

    if len(videos) > 1:
        names = sorted(videos)
        held_out = max(1, round(len(names) * 0.2))
        train = [p for name in names[:-held_out] for p in videos[name]]
        val = [p for name in names[-held_out:] for p in videos[name]]
    else:
        frames = next(iter(videos.values()))
        cut = len(frames) - max(1, round(len(frames) * val_fraction))
        train, val = frames[:cut], frames[cut:]
    return {"train": train, "val": val}


def label_for(image):
    return DATA_DIR / "labels" / f"{image.stem}.txt"


def prepare_dataset(val_fraction):
    images = [
        p for p in (DATA_DIR / "frames").glob("*.jpg") if label_for(p).exists()
    ]
    if not images:
        raise SystemExit("No labeled images. Run label.py first.")
    split = split_images(images, val_fraction)

    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    for part, files in split.items():
        image_dir = DATASET_DIR / "images" / part
        label_dir = DATASET_DIR / "labels" / part
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image in files:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label_for(image), label_dir / label_for(image).name)
        with_fish = sum(bool(label_for(p).read_text().strip()) for p in files)
        print(f"{part}: {len(files)} images, {with_fish} with a fish")
        if not with_fish:
            raise SystemExit(f"The {part} set has no fish. Label more images.")

    config = DATASET_DIR / "data.yaml"
    config.write_text(
        f"path: {json.dumps(DATASET_DIR.as_posix())}\n"
        "train: images/train\nval: images/val\nnames:\n  0: fish\n"
    )
    return config


def overlap(a, b):
    """Intersection over union of two (left, top, right, bottom) boxes."""
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def check(model, image_size):
    """Count hits and false alarms on the held-out images."""
    found = with_fish = false_boxes = without_fish = 0
    for result in model.predict(
        source=str(DATASET_DIR / "images" / "val"),
        imgsz=image_size, conf=0.25, max_det=1,
        save=True, project=str(RUNS_DIR), name="check", stream=True,
    ):
        height, width = result.orig_shape
        label = DATASET_DIR / "labels" / "val" / f"{Path(result.path).stem}.txt"
        truth = []
        for line in label.read_text().splitlines():
            _, cx, cy, w, h = map(float, line.split())
            truth.append(((cx - w / 2) * width, (cy - h / 2) * height,
                          (cx + w / 2) * width, (cy + h / 2) * height))
        guesses = result.boxes.xyxy.tolist()
        if truth:
            with_fish += 1
            found += any(overlap(g, t) >= 0.5 for g in guesses for t in truth)
        else:
            without_fish += 1
            false_boxes += bool(guesses)
    print(f"\nFish found in {found} of {with_fish} held-out images with a fish")
    print(f"False box in {false_boxes} of {without_fish} held-out images without one")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960, help="the fish is small")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--val-fraction", type=float, default=0.3,
        help="share held out when there is only one video",
    )
    args = parser.parse_args()

    config = prepare_dataset(args.val_fraction)
    if args.prepare_only:
        return

    # Never let Ultralytics install packages on its own.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from ultralytics import YOLO, settings

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see the GPU. Check that the CUDA build is installed.")
    settings.update({"sync": False})  # turn off Ultralytics usage analytics

    # Downloaded weights land in the working directory, so keep them under data/.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(MODEL_DIR)

    model = YOLO("yolo26n.pt")
    model.train(
        data=str(config), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=2, seed=0, project=str(RUNS_DIR), name="fish",
    )
    best = model.trainer.best
    print(f"\nBest model: {best}")
    check(YOLO(best), args.imgsz)


if __name__ == "__main__":
    main()
