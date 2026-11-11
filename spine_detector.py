
import logging
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Smaller version of the model used in the GitHub project
model = YOLO("yolo11n-seg.pt")
_ocr_engine: PaddleOCR | None = None
MAX_BOOK_WIDTH_RATIO = 0.3
logger = logging.getLogger(__name__)


def get_ocr_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _ocr_engine


def recognize_book_text(image_crop: np.ndarray, ocr_engine: PaddleOCR) -> str:
    if image_crop.size == 0:
        return ""

    if image_crop.shape[0] > image_crop.shape[1]:
        candidates = [
            cv2.rotate(image_crop, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(image_crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
    else:
        candidates = [image_crop]

    best_text = ""
    best_score = (-1.0, -1)
    for candidate in candidates:
        texts: list[str] = []
        confidence_total = 0.0
        confidence_count = 0
        for result in ocr_engine.predict(input=candidate):
            result_data = result.json
            if callable(result_data):
                result_data = result_data()
            if not isinstance(result_data, dict):
                raise TypeError("PaddleOCR returned an unexpected result format.")

            payload = result_data.get("res", result_data)
            recognized_lines = payload.get("rec_texts", [])
            if recognized_lines is None:
                recognized_lines = []
            confidence_scores = payload.get("rec_scores", [])
            if confidence_scores is None:
                confidence_scores = []
            texts.extend(
                text.strip()
                for text in recognized_lines
                if isinstance(text, str) and text.strip()
            )
            confidence_total += sum(
                float(score) for score in confidence_scores
            )
            confidence_count += len(confidence_scores)

        text = " ".join(texts)
        mean_confidence = (
            confidence_total / confidence_count
            if confidence_count > 0
            else 0.0
        )
        score = (mean_confidence, len(text))
        if score > best_score:
            best_text = text
            best_score = score

    return best_text


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

            width_ratio = (x2 - x1) / image_width
            if width_ratio > MAX_BOOK_WIDTH_RATIO:
                logger.warning(
                    "Skipping broad book detection covering %.0f%% of the image width.",
                    width_ratio * 100,
                )
                continue

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
        book_crop = image[y1:y2, x1:x2]
        book["recognized_text"] = recognize_book_text(
            book_crop, get_ocr_engine()
        )
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
