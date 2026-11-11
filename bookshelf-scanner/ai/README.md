# AI Library project

The AI service is responsible for:

1. Segmenting the bookshelf image using a YOLO model.
2. Extracting and rotating each book spine.
3. Using the Moondream2 LLM model to recognize book titles and authors from each cropped spine image.

## Features

- **YOLO Segmentation**:  
  Detects individual books from a bookshelf image and produces segmented images.

- **Spine Processing**:  
  Crops and rotates book spines to prepare them for text recognition.

- **Moondream2 LLM**:  
  Fine-tuned and quantized model for efficient extraction of book titles and authors from images.

## Prerequisites

- [Python 3.12](https://www.python.org/downloads)
- [Poetry](https://python-poetry.org/docs) for dependency management
- [CUDA Toolkit 12.4 or higher](https://developer.nvidia.com/cuda-toolkit-archive) for GPU acceleration
- The Yolo model will be downloaded automatically during the first run. It will be saved in the `models` directory. The Moondream2 consits of 2 models: text model and multimodal model. The multimodal model will be downloaded automatically during the first run. The text model is quantized and placed in the `/backend/models` directory.

## Installation

1. **Set up the Python environment (optional):**

   ```bash
   poetry config virtualenvs.in-project true
   ```

2. **Install dependencies:**

   ```bash
   poetry install
   ```

## Usage

The backend will send images to the AI service. The AI service will:

1. Segment the image into individual book spines.
2. Process each spine to extract text using the LLM.
3. Return the recognized titles and authors back to the backend.

Also you can include this project as a library in your project. 
Usage if you want to use this project as a library:

```python
from bookscanner_ai import BookPredictor

# Create an instance of the BookPredictor class
book_predictor = BookPredictor()

# Load the models, call this method only once
book_predictor.load_models() 

# Predict the book titles and authors from the image
# The first tuple element is the segmented image as a base64 string
# The second tuple element is the generator of the recognized book titles and authors
segmented_output, results = book_predictor.predict(image_path)

# Display book titles and authors
for result in results:
    print(result)

```
