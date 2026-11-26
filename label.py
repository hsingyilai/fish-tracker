"""Draw a box around the fish in each image (saved in YOLO format).

Drag with the left mouse button to draw a box, then press:
    space  save the box and go to the next image
    n      save "no fish in this image"
    b      go back one image
    s      skip without saving
    q      quit (run again to continue where you left off)
"""

import argparse
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).parent / "data"
WINDOW = "Label the fish"


def to_yolo(corners, width, height):
    """Turn two drag corners into a YOLO line, or None if the box is tiny."""
    (x1, y1), (x2, y2) = corners
    left, right = sorted((min(max(x1, 0), width), min(max(x2, 0), width)))
    top, bottom = sorted((min(max(y1, 0), height), min(max(y2, 0), height)))
    if right - left < 3 or bottom - top < 3:
        return None
    return (
        f"0 {(left + right) / 2 / width:.6f} {(top + bottom) / 2 / height:.6f} "
        f"{(right - left) / width:.6f} {(bottom - top) / height:.6f}\n"
    )


def from_yolo(line, width, height):
    """Corners of a saved box, for showing it again."""
    _, cx, cy, w, h = (float(value) for value in line.split())
    return [
        (round((cx - w / 2) * width), round((cy - h / 2) * height)),
        (round((cx + w / 2) * width), round((cy + h / 2) * height)),
    ]


class Box:
    def __init__(self):
        self.corners = None
        self.dragging = False

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corners, self.dragging = [(x, y), (x, y)], True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.corners[1] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.corners[1], self.dragging = (x, y), False


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    frame_dir = DATA_DIR / "frames"
    label_dir = DATA_DIR / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(frame_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found in {frame_dir}")

    def label_path(image):
        return label_dir / f"{image.stem}.txt"

    index = next(
        (i for i, p in enumerate(images) if not label_path(p).exists()), 0
    )
    box = Box()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, box.mouse)

    while index < len(images):
        image = cv2.imread(str(images[index]))
        if image is None:
            raise SystemExit(f"Could not read {images[index]}")
        height, width = image.shape[:2]
        target = label_path(images[index])
        saved = target.read_text().strip() if target.exists() else None
        box.corners = from_yolo(saved, width, height) if saved else None
        box.dragging = False

        while True:
            view = image.copy()
            if box.corners:
                cv2.rectangle(view, box.corners[0], box.corners[1], (0, 255, 0), 2)
            status = f"{index + 1}/{len(images)}"
            if saved == "":
                status += "  saved: no fish"
            cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(20) & 0xFF

            if key == ord(" ") and box.corners:
                line = to_yolo(box.corners, width, height)
                if line:
                    target.write_text(line)
                    index += 1
                    break
            elif key == ord("n"):
                target.write_text("")
                index += 1
                break
            elif key == ord("s"):
                index += 1
                break
            elif key == ord("b"):
                index = max(0, index - 1)
                break
            elif key == ord("q"):
                index = len(images)
                break

    cv2.destroyAllWindows()
    done = sum(label_path(p).exists() for p in images)
    print(f"Labeled {done} of {len(images)} images. Labels are in {label_dir}")


if __name__ == "__main__":
    main()
