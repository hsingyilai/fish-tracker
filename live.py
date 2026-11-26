"""Show the camera live, with a box where the model finds the fish.

    python live.py               # newest model; press q to quit
    python live.py --conf 0.1    # also show weaker guesses
    python live.py --model data/runs/fish-4/weights/best.pt

Nothing is saved.
"""

import argparse
import os
import time
from pathlib import Path

import cv2

import record
from train import RUNS_DIR

WINDOW = "Live fish detector (press q to quit)"
GREEN, YELLOW = (0, 255, 0), (0, 255, 255)


def newest_model():
    models = sorted(
        RUNS_DIR.glob("fish*/weights/best.pt"), key=lambda p: p.stat().st_mtime
    )
    if not models:
        raise SystemExit("No trained model found. Run train.py first.")
    return models[-1]


def write(frame, text, position, colour):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)


def watch(camera, model, imgsz, conf):
    """Detect and display until q is pressed or the window is closed."""
    previous, rate = time.monotonic(), 0.0
    while True:
        frame = record.read_frame(camera)
        boxes = model.predict(
            frame, imgsz=imgsz, conf=conf, max_det=1, verbose=False
        )[0].boxes

        status = "no fish found"
        for box, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
            left, top, right, bottom = (round(v) for v in box)
            status = f"fish {score:.2f}"
            cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)
            write(frame, status, (left, max(20, top - 8)), GREEN)

        now = time.monotonic()
        current = 1 / max(now - previous, 1e-6)
        rate = current if rate == 0 else 0.9 * rate + 0.1 * current
        previous = now
        write(frame, f"{status}   {rate:.0f} frames/s", (10, 30), YELLOW)

        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break  # closed with the X button


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, help="default: the newest best.pt")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence cutoff")
    parser.add_argument("--imgsz", type=int, default=960, help="as used in training")
    args = parser.parse_args()

    path = args.model or newest_model()
    if not path.exists():
        raise SystemExit(f"No model at {path}")

    # Never let Ultralytics install packages on its own.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    print(f"Using {path}")
    model = YOLO(str(path))
    camera = None
    try:
        camera = record.open_camera(args.camera)
        watch(camera, model, args.imgsz, args.conf)
    except record.CameraError as error:
        raise SystemExit(str(error))
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
