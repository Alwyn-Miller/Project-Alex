
from camera import capture_bookshelf
from spine_detector import detect_books


def main():
    print("HRI Smart Bookshelf")

    # Capture bookshelf image
    image_path = capture_bookshelf()

    if image_path is None:
        print("No image captured.")
        return

    # Detect books
    books = detect_books(image_path)

    print(f"Detected {len(books)} books")

    for book in books:
        print(book)


if __name__ == "__main__":
    main()
