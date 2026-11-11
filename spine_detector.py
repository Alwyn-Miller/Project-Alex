
import cv2
from ultralytics import YOLO
from pathlib import Path

# Smaller version of the model used in the GitHub project
model = YOLO("yolo11n-seg.pt")


def calculate_book_size_and_location(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> dict[str, int | float | str]:
    """Return image-pixel size and frame-relative location, not physical measurements."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if x2 < x1 or y2 < y1:
        raise ValueError("Bounding box coordinates are invalid.")

    width = x2 - x1
    height = y2 - y1
    centre_x = (x1 + x2) / 2
    centre_y = (y1 + y2) / 2
    relative_x = centre_x / image_width
    relative_y = centre_y / image_height

    return {
        "width_px": width,
        "height_px": height,
        "centre_x_px": centre_x,
        "centre_y_px": centre_y,
        "relative_width": width / image_width,
        "relative_height": height / image_height,
        "relative_x": relative_x,
        "relative_y": relative_y,
        "horizontal_position": (
            "left" if relative_x < 1 / 3 else "centre" if relative_x < 2 / 3 else "right"
        ),
        "vertical_position": (
            "top" if relative_y < 1 / 3 else "middle" if relative_y < 2 / 3 else "bottom"
        ),
    }


def detect_books(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    image_height, image_width = image.shape[:2]

    # COCO class 73 = book
    results = model.predict(
        source=image,
        classes=[73],
        conf=0.25,
        verbose=False
    )

    output = image.copy()
    books = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])
            measurements = calculate_book_size_and_location(
                x1, y1, x2, y2, image_width, image_height
            )

            book = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                **measurements,
                "confidence": confidence,
            }

            books.append(book)

    # Sort detected books from left to right
    books.sort(key=lambda book: book["centre_x_px"])
    for book_id, book in enumerate(books, start=1):
        book["book_id"] = book_id
        x1, y1, x2, y2 = book["x1"], book["y1"], book["x2"], book["y2"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"Book {book_id}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    Path("output").mkdir(exist_ok=True)

    cv2.imwrite(
        "output/detected_books.jpg",
        output
    )

    return books
