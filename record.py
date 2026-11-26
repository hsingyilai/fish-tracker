"""Record short clips of the tank from a webcam.

Examples:
    python record.py --preview               # aim the camera, press q to quit
    python record.py                         # one 60 s clip
    python record.py --clips 12 --every 3600 # one clip per hour, 12 times
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

VIDEO_DIR = Path(__file__).parent / "data" / "videos"


def open_camera(index):
    # On Windows the default backend (Media Foundation) was unreliable with this
    # USB camera, so use DirectShow there.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {index}. Try --camera 1.")
    return camera


def read_frame(camera):
    ok, frame = camera.read()
    if not ok:
        raise SystemExit("The camera stopped sending frames.")
    return frame


def preview(index):
    camera = open_camera(index)
    try:
        while True:
            cv2.imshow("Preview (press q to quit)", read_frame(camera))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def record_clip(index, seconds, fps):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    path = VIDEO_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
    camera = open_camera(index)
    writer = None
    try:
        frame = read_frame(camera)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise SystemExit(f"Could not create {path}")

        start = time.monotonic()
        written = 0
        while written < seconds * fps:
            frame = read_frame(camera)
            # Keep playback in real time even if the camera's rate differs.
            due = min(int((time.monotonic() - start) * fps), int(seconds * fps))
            while written < due:
                writer.write(frame)
                written += 1
    finally:
        camera.release()
        if writer is not None:
            writer.release()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--seconds", type=float, default=60, help="clip length")
    parser.add_argument("--fps", type=float, default=15, help="saved frame rate")
    parser.add_argument("--clips", type=int, default=1, help="number of clips")
    parser.add_argument(
        "--every", type=float, default=3600, help="seconds between clip starts"
    )
    parser.add_argument("--preview", action="store_true", help="only show the view")
    args = parser.parse_args()

    if args.preview:
        preview(args.camera)
        return

    for number in range(args.clips):
        start = time.monotonic()
        record_clip(args.camera, args.seconds, args.fps)
        if number < args.clips - 1:
            time.sleep(max(0, args.every - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
