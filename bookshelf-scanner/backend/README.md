# Backend Service

The backend service is a FastAPI application that receives bookshelf images from the frontend, sends them to the AI service for processing, and streams back the segmented image and recognized book titles/authors to the client.

## Features

- **API Endpoint (`POST /api/predict`)**:  
  Accepts an uploaded bookshelf image and returns:
  - A segmented image of the bookshelf, showing identified book spines.
  - A stream of recognized titles and authors for each detected book.

- **Asynchronous Processing**:  
  Utilizes Python's async capabilities for efficient I/O and streaming responses.

- **Easy Integration**:  
  Designed to work seamlessly with the frontend and AI components.

## Prerequisites

- [Python 3.12](https://www.python.org/downloads)
- [Poetry](https://python-poetry.org/docs) for dependency management.

## Installation

1. **Set up the Python environment (optional but recommended):**

   ```bash
   poetry config virtualenvs.in-project true
   ```

2. **Install dependencies:**

   ```bash
   poetry install
   ```

## Running the Backend

Use the following command to start the FastAPI server in development mode:

```bash
poetry run fastapi dev src/main.py
```

The API should be available at `http://localhost:8000/docs`. You can use this endpoint to:

- Explore the API routes.
- Test the `/api/predict` endpoint by uploading an image.
