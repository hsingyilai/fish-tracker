"""Score a trained model on videos it was not trained on.

    python evaluate.py 20261126          # every video whose name starts like that
    python evaluate.py 20261126_105333   # one video
    python evaluate.py 20261126 --model data/runs/fish/weights/best.pt

Extract and label the videos first. Add --save-mistakes to keep pictures of what
the model got wrong, with your label in green and its box in red.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2

from live import newest_model
from train import RUNS_DIR, TRAINED_ON, label_for, labeled_images

OUTCOMES = ("found", "missed", "off_target", "quiet", "false_box")
MISTAKES = ("missed", "off_target", "false_box")


def trained_videos(model_path):
    """The videos the model learned from, as listed by train.py."""
    listing = model_path.resolve().parent.parent / TRAINED_ON
    if not listing.exists():
        raise SystemExit(
            f"Cannot tell which videos {model_path} was trained on "
            f"({TRAINED_ON} is missing). Train a model with train.py."
        )
    return set(listing.read_text().splitlines())


def read_boxes(label, width, height):
    boxes = []
    for line in label.read_text().splitlines():
        _, cx, cy, w, h = map(float, line.split())
        boxes.append(((cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def overlap(a, b):
    """Intersection over union of two (left, top, right, bottom) boxes."""
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def judge(truths, guesses):
    """What happened on one image: a box counts if it overlaps by half or more."""
    if not truths:
        return "false_box" if guesses else "quiet"
    if any(overlap(g, t) >= 0.5 for g in guesses for t in truths):
        return "found"
    return "off_target" if guesses else "missed"


def report(name, counts):
    fish = counts["found"] + counts["missed"] + counts["off_target"]
    empty = counts["quiet"] + counts["false_box"]
    print(
        f"{name}: fish found in {counts['found']} of {fish} "
        f"({counts['missed']} missed, {counts['off_target']} off target); "
        f"false box in {counts['false_box']} of {empty} without fish"
    )


def save_mistake(path, image, truths, guesses):
    for boxes, colour in ((truths, (0, 255, 0)), (guesses, (0, 0, 255))):
        for left, top, right, bottom in boxes:
            cv2.rectangle(image, (round(left), round(top)),
                          (round(right), round(bottom)), colour, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+", help="video names, or how they start")
    parser.add_argument("--model", type=Path, help="default: the newest best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence cutoff")
    parser.add_argument("--imgsz", type=int, default=960, help="as used in training")
    parser.add_argument("--save-mistakes", action="store_true")
    args = parser.parse_args()

    model_path = args.model or newest_model()
    if not model_path.exists():
        raise SystemExit(f"No model at {model_path}")
    videos = labeled_images(args.videos)
    used = sorted(set(videos) & trained_videos(model_path))
    if used:
        raise SystemExit(
            f"This model was trained on: {', '.join(used)}. Name only other videos."
        )

    # Never let Ultralytics install packages on its own.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    print(f"Model: {model_path}")
    model = YOLO(str(model_path))
    run = model_path.resolve().parent.parent.name
    mistake_dir = RUNS_DIR / "eval" / f"{run}_{datetime.now():%Y%m%d_%H%M%S}"
    total = dict.fromkeys(OUTCOMES, 0)

    for name, images in videos.items():
        counts = dict.fromkeys(OUTCOMES, 0)
        for path in images:
            image = cv2.imread(str(path))
            height, width = image.shape[:2]
            truths = read_boxes(label_for(path), width, height)
            guesses = model.predict(
                image, imgsz=args.imgsz, conf=args.conf, max_det=1, verbose=False
            )[0].boxes.xyxy.tolist()
            outcome = judge(truths, guesses)
            counts[outcome] += 1
            if args.save_mistakes and outcome in MISTAKES:
                save_mistake(mistake_dir / f"{outcome}_{path.name}",
                             image, truths, guesses)
        report(name, counts)
        for key in OUTCOMES:
            total[key] += counts[key]

    if len(videos) > 1:
        report("All", total)
    if mistake_dir.exists():
        print(f"Mistakes saved in {mistake_dir}")


if __name__ == "__main__":
    main()
