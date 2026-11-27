"""Score a trained model on labeled images, without training.

    python evaluate.py                   # videos recorded after the model was trained
    python evaluate.py 20261127_101500   # the videos you name
    python evaluate.py --model data/runs/fish-4/weights/best.pt 20261127_101500

Extract and label the videos first. Images the model got wrong are saved for
review, with your label in green and the model's box in red.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2

from live import newest_model
from train import DATA_DIR, RUNS_DIR, label_for, overlap

OUTCOMES = ("found", "missed", "off_target", "quiet", "false_box")
MISTAKES = ("missed", "off_target", "false_box")


def recorded_after(video, moment):
    """True if the video's name is a time later than `moment`."""
    try:
        return datetime.strptime(video, "%Y%m%d_%H%M%S") > moment
    except ValueError:
        return False  # a renamed or phone video: its time is unknown


def read_boxes(label, width, height):
    boxes = []
    for line in label.read_text().splitlines():
        _, cx, cy, w, h = map(float, line.split())
        boxes.append(((cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def judge(truths, guesses):
    """What happened on one image, using the same rule as train.py."""
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
    parser.add_argument("videos", nargs="*", help="video names, e.g. 20261127_101500")
    parser.add_argument("--model", type=Path, help="default: the newest best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence cutoff")
    parser.add_argument("--imgsz", type=int, default=960, help="as used in training")
    args = parser.parse_args()

    model_path = args.model or newest_model()
    if not model_path.exists():
        raise SystemExit(f"No model at {model_path}")
    trained = datetime.fromtimestamp(model_path.stat().st_mtime)

    images = {}
    for image in sorted((DATA_DIR / "frames").glob("*.jpg")):
        images.setdefault(image.stem.rsplit("_", 1)[0], []).append(image)

    names = [Path(v).stem for v in args.videos] or [
        v for v in images if recorded_after(v, trained)
    ]
    if not names:
        raise SystemExit(
            f"No images from videos recorded after this model was trained "
            f"({trained:%Y-%m-%d %H:%M}). Record, extract and label one, "
            f"or name a video."
        )
    unknown = [n for n in names if n not in images]
    if unknown:
        raise SystemExit(f"No images for: {', '.join(unknown)}")

    # Never let Ultralytics install packages on its own.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    run = model_path.resolve().parent.parent.name
    print(f"Model: {model_path}")
    model = YOLO(str(model_path))
    mistake_dir = RUNS_DIR / "eval" / f"{run}_{datetime.now():%Y%m%d_%H%M%S}"
    total = dict.fromkeys(OUTCOMES, 0)

    for name in names:
        if not recorded_after(name, trained):
            print(f"Warning: {name} is not newer than the model, which may "
                  f"have used it. Do not report this as a test score.")
        labeled = [p for p in images[name] if label_for(p).exists()]
        if len(labeled) < len(images[name]):
            print(f"{name}: skipping {len(images[name]) - len(labeled)} unlabeled images")
        if not labeled:
            continue

        counts = dict.fromkeys(OUTCOMES, 0)
        for path in labeled:
            image = cv2.imread(str(path))
            height, width = image.shape[:2]
            truths = read_boxes(label_for(path), width, height)
            guesses = model.predict(
                image, imgsz=args.imgsz, conf=args.conf, max_det=1, verbose=False
            )[0].boxes.xyxy.tolist()
            outcome = judge(truths, guesses)
            counts[outcome] += 1
            if outcome in MISTAKES:
                save_mistake(mistake_dir / f"{outcome}_{path.name}",
                             image, truths, guesses)
        report(name, counts)
        for key in OUTCOMES:
            total[key] += counts[key]

    if not sum(total.values()):
        raise SystemExit("Nothing labeled to score. Run label.py first.")
    if len(names) > 1:
        report("All", total)
    if mistake_dir.exists():
        print(f"Mistakes saved in {mistake_dir}")


if __name__ == "__main__":
    main()
