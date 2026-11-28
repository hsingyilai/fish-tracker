"""Train a fish detector on the videos you name.

    python train.py 20261127_09 20261127_1   # video names, or how they start

Every labeled image from those videos is used for training, and nothing is held
out. To find out how good the model is, run evaluate.py on other videos.
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
TRAINED_ON = "trained_on.txt"  # the list of videos, kept in each run folder


def label_for(image):
    return DATA_DIR / "labels" / f"{image.stem}.txt"


def labeled_images(wanted):
    """The labeled images of the named videos, as {video name: [images]}.

    Each name may be a whole video name or just its beginning.
    """
    videos = {}
    for image in sorted((DATA_DIR / "frames").glob("*.jpg")):
        videos.setdefault(image.stem.rsplit("_", 1)[0], []).append(image)

    chosen = set()
    for name in wanted:
        matches = [v for v in videos if v.startswith(Path(name).stem)]
        if not matches:
            raise SystemExit(f"No images from a video starting with {name}")
        chosen.update(matches)

    result = {}
    for video in sorted(chosen):
        labeled = [p for p in videos[video] if label_for(p).exists()]
        if len(labeled) < len(videos[video]):
            print(f"{video}: skipping {len(videos[video]) - len(labeled)} unlabeled images")
        if labeled:
            result[video] = labeled
    if not result:
        raise SystemExit("None of those images are labeled. Run label.py first.")
    return result


def prepare_dataset(videos):
    """Copy the images and labels into the layout Ultralytics expects."""
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    image_dir = DATASET_DIR / "images" / "train"
    label_dir = DATASET_DIR / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    total = with_fish = 0
    for video, images in videos.items():
        fish = sum(bool(label_for(p).read_text().strip()) for p in images)
        print(f"{video}: {len(images)} images, {fish} with a fish")
        total += len(images)
        with_fish += fish
        for image in images:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label_for(image), label_dir / label_for(image).name)
    print(f"Training on {len(videos)} videos: {total} images, {with_fish} with a fish")
    if not with_fish:
        raise SystemExit("There is no fish in these images. Name other videos.")

    # Ultralytics insists on a "val" entry. It points at the training images,
    # and training runs with val=False, so nothing is chosen by validation.
    config = DATASET_DIR / "data.yaml"
    config.write_text(
        f"path: {json.dumps(DATASET_DIR.as_posix())}\n"
        "train: images/train\nval: images/train\nnames:\n  0: fish\n"
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+", help="video names, or how they start")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960, help="the fish is small")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    videos = labeled_images(args.videos)
    config = prepare_dataset(videos)

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
        val=False,
    )
    run = Path(model.trainer.save_dir)
    (run / TRAINED_ON).write_text("".join(f"{video}\n" for video in videos))
    print(f"\nModel: {model.trainer.best}")
    print("Any scores printed above come from the training images: ignore them.")
    print("Score the model on other videos with: python evaluate.py <video names>")


if __name__ == "__main__":
    main()
