"""Record short clips of the tank from a webcam.

Examples:
    python record.py --preview               # aim the camera, press q to quit
    python record.py                         # one 60 s clip
    python record.py --clips 12 --every 3600 # one clip per hour, 12 times

If the camera fails, recording is retried. A clip that still cannot be recorded
is skipped, and the later clips go ahead.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

VIDEO_DIR = Path(__file__).parent / "data" / "videos"
RETRIES = 5  # failures in a row before giving up on a clip
RETRY_WAIT = 10  # seconds


class CameraError(Exception):
    frames = 0  # frames saved before the failure


def open_camera(index):
    # On Windows the default backend (Media Foundation) was unreliable with this
    # USB camera, so use DirectShow there.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise CameraError(f"Could not open camera {index}.")
    return camera


def read_frame(camera):
    ok, frame = camera.read()
    if not ok:
        raise CameraError("The camera stopped sending frames.")
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


def record_clip(index, frame_count, fps):
    """Record one video file of `frame_count` frames."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    path = VIDEO_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
    camera = writer = None
    written = 0
    try:
        camera = open_camera(index)
        frame = read_frame(camera)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise SystemExit(f"Could not create {path}")

        start = time.monotonic()
        while written < frame_count:
            frame = read_frame(camera)
            # Keep playback in real time even if the camera's rate differs.
            due = min(int((time.monotonic() - start) * fps), frame_count)
            while written < due:
                writer.write(frame)
                written += 1
    except CameraError as error:
        error.frames = written
        raise
    finally:
        if camera is not None:
            camera.release()
        if writer is not None:
            writer.release()
            if written:
                print(f"Saved {path} ({written / fps:.1f} s)")
            else:
                path.unlink(missing_ok=True)  # don't leave an empty file


def record_with_retries(index, seconds, fps):
    """Record `seconds` of video, retrying if the camera fails.

    Footage saved before a failure is kept, and only the rest is recorded, in a
    new file. Returns False after too many failures in a row.
    """
    remaining = round(seconds * fps)
    failures = 0
    while True:
        try:
            record_clip(index, remaining, fps)
            return True
        except CameraError as error:
            print(error)
            remaining -= error.frames
            if error.frames >= 5 * fps:  # it worked for a while: count afresh
                failures = 0
            failures += 1
            if failures > RETRIES:
                return False
            print(f"Retry {failures} of {RETRIES} in {RETRY_WAIT} s")
            time.sleep(RETRY_WAIT)


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
        try:
            preview(args.camera)
        except CameraError as error:
            raise SystemExit(f"{error} Try --camera 1.")
        return

    failed = 0
    for number in range(args.clips):
        start = time.monotonic()
        print(f"Clip {number + 1} of {args.clips}")
        if not record_with_retries(args.camera, args.seconds, args.fps):
            failed += 1
            print(f"Gave up on clip {number + 1}")
        if number < args.clips - 1:
            time.sleep(max(0, args.every - (time.monotonic() - start)))
    if failed:
        raise SystemExit(f"{failed} of {args.clips} clips were not completed.")


if __name__ == "__main__":
    main()
