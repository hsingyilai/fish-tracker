"""Save still images from the recorded videos, ready for labeling.

New videos go in data/videos/. Once its images are saved, a video is moved to
data/videos/extracted/, so it is not processed again. To extract it again, move
it back.

Example:
    python extract_frames.py --interval 2    # one image every 2 s of video
"""

import argparse
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).parent / "data"
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv"}


def find_videos(folder):
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in VIDEO_TYPES)


def extract(video_path, frame_dir, interval):
    """Save images from one video and return how many were saved."""
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            return 0
        fps = video.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(fps * interval))
        index = saved = 0
        while True:
            ok, frame = video.read()
            if not ok:
                return saved
            if index % step == 0:
                target = frame_dir / f"{video_path.stem}_{index:06d}.jpg"
                if not cv2.imwrite(str(target), frame):
                    raise SystemExit(f"Could not write {target}")
                saved += 1
            index += 1
    finally:
        video.release()  # Windows cannot move a file that is still open


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--interval", type=float, default=2, help="seconds between saved images"
    )
    args = parser.parse_args()

    video_dir = DATA_DIR / "videos"
    done_dir = video_dir / "extracted"
    frame_dir = DATA_DIR / "frames"
    done_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(video_dir)
    if not videos:
        print(f"No new videos in {video_dir}")
        return

    total = moved = 0
    for video in videos:
        # Images are named after their video, so a repeated name would overwrite
        # another video's images, and the labels would no longer match them.
        if any(v.stem.lower() == video.stem.lower() for v in find_videos(done_dir)):
            print(f"Left {video.name}: this name was already extracted. Rename it.")
            continue
        saved = extract(video, frame_dir, args.interval)
        if not saved:
            print(f"Left {video.name}: could not read it")
            continue
        total += saved
        try:
            video.rename(done_dir / video.name)
            moved += 1
        except OSError as error:
            print(f"Extracted {video.name} but could not move it: {error}")
    print(f"Saved {total} images to {frame_dir}")
    print(f"Moved {moved} videos to {done_dir}")


if __name__ == "__main__":
    main()
