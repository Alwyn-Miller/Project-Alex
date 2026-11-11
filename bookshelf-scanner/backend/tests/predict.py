from bookscanner_ai import BookPredictor

def test_predict() -> None:
    book_predictor = BookPredictor()
    book_predictor.load_models()
    result = book_predictor.predict("../ai/dataset/images/img_1.jpg")

    if not result:
        print("No books detected.")
        return
    
    for book in result[1]:
        print(book)

if __name__ == "__main__":
    test_predict()
