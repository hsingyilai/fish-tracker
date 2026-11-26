"""Save still images from the recorded videos, ready for labeling.

Example:
    python extract_frames.py --interval 2    # one image every 2 s of video
"""

import argparse
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).parent / "data"
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv"}


def extract(video_path, frame_dir, interval):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print(f"Skipped {video_path.name}: could not open it")
        return 0

    fps = video.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps * interval))
    index = saved = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if index % step == 0:
            target = frame_dir / f"{video_path.stem}_{index:06d}.jpg"
            if not cv2.imwrite(str(target), frame):
                raise SystemExit(f"Could not write {target}")
            saved += 1
        index += 1
    video.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--interval", type=float, default=2, help="seconds between saved images"
    )
    args = parser.parse_args()

    video_dir = DATA_DIR / "videos"
    frame_dir = DATA_DIR / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in video_dir.glob("*") if p.suffix.lower() in VIDEO_TYPES
    )
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")

    total = sum(extract(video, frame_dir, args.interval) for video in videos)
    print(f"Saved {total} images to {frame_dir}")


if __name__ == "__main__":
    main()
