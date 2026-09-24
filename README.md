# Visual Product Matcher for Home-Design Catalogs

A lightweight visual search service that retrieves visually similar furniture and home-decor products from a catalog based on an uploaded image. Built with PyTorch, OpenCV, Pillow, FastAPI, and PostgreSQL with the pgvector extension.

---

## Overview

In home design and e-commerce platforms, text-based search often struggles with visual attributes like furniture geometry, texture, and aesthetic style. This project implements a visual similarity search pipeline:
1. Accepts an uploaded product image via a REST API.
2. Validates and preprocesses the image using Pillow and OpenCV.
3. Extracts a 512-dimensional visual feature embedding using a pretrained PyTorch ResNet-18 model.
4. Performs an approximate nearest neighbor (ANN) search using cosine distance in PostgreSQL via pgvector.
5. Returns ranked matching products with metadata and similarity scores.

---

## Problem Statement

Home-design catalogs contain items where visual appearance (color palette, shape, material finish, arm style on a chair or sofa) is difficult for users to describe in keywords. 
- Traditional text search requires exact keyword matches or extensive manual tagging.
- Maintaining separate dedicated vector databases introduces operational complexity, data synchronization lag, and extra infrastructure costs for small-to-medium catalogs.

This project solves this by combining relational product metadata and high-dimensional image embeddings in a single PostgreSQL database using pgvector, providing fast, filtered visual matching through a REST API.

---

## Key Features

- **Visual Similarity Search**: Upload an image to find the top-K visually similar furniture items in the catalog.
- **Category Filtering**: Combine visual search with SQL category filters (e.g., search only within sofa or chair).
- **Input Validation & Safety**: Validates file format (JPEG, PNG, WebP), byte integrity, and limits file upload size (default: 5 MB).
- **Aspect-Ratio Preserving Preprocessing**: Uses OpenCV letterboxing with neutral gray padding to prevent object distortion during resizing.
- **L2-Normalized Embeddings**: Ensures unit-length feature vectors so that cosine similarity aligns with dot-product distance.
- **Relational + Vector Storage**: Stores product metadata (price, dimensions, brand, material) alongside vectors in PostgreSQL.
- **HNSW Indexing**: Uses Hierarchical Navigable Small World graph indexing for fast vector retrieval.
- **Automated Test Suite**: 16 unit and integration tests covering image validation, model feature extraction, database queries, and API error responses.

---

## Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core application and script development |
| **Web Framework** | FastAPI | REST API endpoints, request validation, and OpenAPI documentation |
| **Vision Backbone** | PyTorch and TorchVision | Pretrained ResNet-18 feature extraction (512-dimensional embeddings) |
| **Image Preprocessing** | OpenCV (`cv2`) | Aspect-ratio letterboxing, resizing, and array transformations |
| **Image Handling** | Pillow (`PIL`) | Image decoding, format verification, and EXIF orientation correction |
| **Database** | PostgreSQL 16 | Relational product metadata storage |
| **Vector Search** | pgvector | Vector data type, HNSW indexing, and cosine distance (`<=>`) operator |
| **ORM** | SQLAlchemy | Database schema mapping and query construction |
| **Testing** | pytest and HTTPX | Automated unit and integration testing |
| **Containerization** | Docker and Docker Compose | Local PostgreSQL + pgvector environment setup |

---

## System Architecture / Workflow

The system follows a synchronous request-response flow:

```
User / Client
     |
     v (Uploads image: JPEG/PNG/WebP, max 5MB)
FastAPI Endpoint (POST /search)
     |
     v (Raw bytes)
Preprocessing Pipeline (Pillow: validation and EXIF -> OpenCV: letterbox to 224x224 and normalization)
     |
     v (Preprocessed Tensor: [1, 3, 224, 224])
PyTorch Model (Pretrained ResNet-18 -> 512-dim L2-normalized vector)
     |
     v (512-dim embedding vector)
PostgreSQL + pgvector (HNSW index query using <=> cosine distance)
     |
     v (Ranked product records + distances)
JSON Response (Product metadata + similarity scores)
```

---

## Machine Learning & Vision Pipeline

### 1. Dataset
- A structured catalog of home-design furniture products defined in `data/catalog_seed.json`.
- Categories include: `sofa`, `chair`, `table`, `bed`, `cabinet`, `lamp`.
- Realistic metadata: name, category, brand, price, dimensions, material, color, image path, description.
- Offline sample images generated via `scripts/generate_sample_images.py` for reproducible local testing.

### 2. Preprocessing Pipeline (`app/core/preprocessor.py`)
- **Format & Byte Verification**: Pillow opens and verifies file headers, rejecting non-image binaries or corrupted payloads.
- **EXIF Transposition**: Corrects image orientation for photos taken on mobile devices.
- **Channel Normalization**: Converts RGBA, grayscale, or CMYK images to standard 3-channel RGB.
- **Letterbox Resizing**: Uses OpenCV to resize the image to fit within a 224x224 canvas while preserving the original aspect ratio, filling remaining space with neutral gray (`(128, 128, 128)`). This avoids artificial stretching of furniture silhouettes.
- **Standardization**: Scales pixel values to `[0.0, 1.0]` and standardizes using ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).

### 3. Feature Engineering & Model Architecture (`app/core/vision_model.py`)
- **Model Backbone**: Pretrained ResNet-18 convolutional neural network.
- **Feature Extraction**: The final classification layer (`model.fc`) is replaced with `nn.Identity()`.
- **Embedding Generation**: Passing a `(1, 3, 224, 224)` tensor through the network yields a 512-dimensional dense feature vector.
- **L2 Normalization**: Vectors are divided by their Euclidean norm (unit-norm, `||v|| = 1.0`).
- **Inference Mode**: Inference is executed under `torch.no_grad()` to prevent memory allocation for autograd computation graphs.

### 4. Training vs. Pretrained Inference
- This project utilizes transfer learning via pretrained ImageNet weights. The network acts as a fixed feature extractor without requiring costly retraining from scratch, which is suitable for general visual similarity matching on standard objects.

### 5. Framework Comparison Experiment (`scripts/compare_frameworks.py`)
To evaluate framework characteristics, a benchmark script compares PyTorch ResNet-18 against TensorFlow/Keras MobileNetV2:
- **PyTorch (ResNet-18)**: Produces 512-dimensional embeddings, resulting in smaller index sizes in pgvector and average CPU inference latency of ~40 ms.
- **TensorFlow (MobileNetV2)**: Produces 1280-dimensional embeddings with average CPU inference latency of ~50 ms.
- **Conclusion**: PyTorch ResNet-18 was selected for the primary API due to the compact embedding size (saving memory in pgvector HNSW graphs) and low runtime overhead.

---

## Project Structure

```
Visual-Product-Matcher-for-Home-Design-Catalogs/
|-- app/
|   |-- __init__.py
|   |-- main.py                     # FastAPI app factory, CORS, exception handlers
|   |-- config.py                   # Pydantic Settings and environment configuration
|   |-- api/
|   |   |-- __init__.py
|   |   `-- routes.py               # API endpoint implementations
|   |-- core/
|   |   |-- __init__.py
|   |   |-- preprocessor.py         # Pillow + OpenCV image preprocessing
|   |   `-- vision_model.py         # PyTorch feature extractor
|   |-- db/
|   |   |-- __init__.py
|   |   |-- session.py              # Database connection and session lifecycle
|   |   `-- models.py               # SQLAlchemy Product model with Vector(512)
|   `-- schemas/
|       |-- __init__.py
|       `-- product.py              # Pydantic request and response schemas
|-- data/
|   |-- catalog_seed.json           # Catalog seed data with product metadata
|   `-- sample_images/              # Generated sample images for catalog items
|-- docker/
|   `-- docker-compose.yml          # PostgreSQL 16 + pgvector container definition
|-- scripts/
|   |-- generate_sample_images.py   # Generates sample images for catalog items
|   |-- seed_catalog.py             # Seeds PostgreSQL database and computes embeddings
|   `-- compare_frameworks.py       # Benchmark comparing PyTorch and TensorFlow
|-- sql/
|   `-- schema.sql                  # PostgreSQL table creation and HNSW index DDL
|-- tests/
|   |-- __init__.py
|   |-- conftest.py                 # Pytest fixtures, mock vision model, SQLite test DB
|   |-- test_api.py                 # API integration tests
|   |-- test_database.py            # Model and similarity math tests
|   `-- test_vision.py              # Preprocessing and embedding unit tests
|-- .env.example                    # Example environment variables
|-- .gitignore                      # Git ignore file
|-- requirements.txt                # Python package dependencies
`-- README.md                       # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Docker and Docker Compose (or a local PostgreSQL instance with pgvector)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/harsharaju1314-hash/Visual-Product-Matcher-for-Home-Design-Catalogs.git
cd Visual-Product-Matcher-for-Home-Design-Catalogs
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scriptsctivate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start PostgreSQL with pgvector
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

---

## Running the Application

### 1. Seed the Database
Generate sample catalog images and populate the database with product metadata and embeddings:
```bash
python scripts/generate_sample_images.py
python scripts/seed_catalog.py
```

### 2. Start the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive Swagger API documentation is available at:
**http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check (DB connection, model status, catalog count) |
| `POST` | `/search` | Visual similarity search by image upload (with optional `category` and `top_k`) |
| `GET` | `/products/{product_id}` | Retrieve single product metadata by ID |
| `POST` | `/products` | Register a new product into the catalog (optional admin route) |

---

## Database Schema & Indexing

### Schema Definition (`sql/schema.sql`)
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    brand VARCHAR(100),
    price NUMERIC(10, 2) NOT NULL,
    dimensions VARCHAR(100),
    material VARCHAR(100),
    color VARCHAR(50),
    image_url TEXT NOT NULL,
    description TEXT,
    embedding vector(512),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_products_embedding_hnsw 
ON products USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_products_category 
ON products (category);
```

### Why HNSW Indexing?
- **Graph-Based Navigation**: HNSW constructs a multi-layer graph where upper layers contain long-range connections and lower layers contain fine connections.
- **Logarithmic Complexity**: Enables approximate nearest neighbor search in O(log N) time complexity.
- **Cosine Distance (`<=>`)**: For unit-normalized vectors, cosine distance is computed as `1.0 - cosine_similarity`.

---

## Testing

The test suite runs with `pytest` using an in-memory SQLite database and deterministic mock fixtures so it can execute in any environment without external services.

Run the test suite:
```bash
pytest -v
```

### Test Coverage (16 Tests):
- **Image Preprocessing (`tests/test_vision.py`)**:
  - Valid JPEG preprocessing and output tensor dimensions `(1, 3, 224, 224)`.
  - RGBA to RGB conversion.
  - Corrupted byte detection and empty byte handling.
  - PyTorch 512-dim embedding extraction and L2 unit-norm check.
- **Database Logic (`tests/test_database.py`)**:
  - Product model column and metadata mappings.
  - Cosine distance to similarity score conversion.
- **API Endpoints (`tests/test_api.py`)**:
  - `GET /health` response and metadata verification.
  - `GET /products/{id}` lookup and 404 handling.
  - `POST /products` item creation.
  - `POST /search` visual search response formatting.
  - 400 Bad Request on unsupported file extensions (`.txt`).
  - 400 Bad Request on corrupted image bytes.
  - 413 Content Too Large on files > 5 MB.
  - 422 Unprocessable Entity when file parameter is missing.

---

## Sample Usage

### 1. Perform a Visual Search
```bash
curl -X POST "http://localhost:8000/search?top_k=2&category=chair"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@data/sample_images/query_yellow_chair.jpg"
```

**Response:**
```json
{
  "total_results": 2,
  "top_k": 2,
  "query_image_info": {
    "format": "JPEG",
    "dimensions": "400x400",
    "file_size_bytes": 18240
  },
  "results": [
    {
      "id": 4,
      "name": "Mid-Century Modern Accent Lounge Chair - Mustard Yellow",
      "category": "chair",
      "brand": "Moderna",
      "price": 279.0,
      "dimensions": "30W x 31D x 33H in",
      "material": "Textured Tweed & Walnut",
      "color": "Mustard Yellow",
      "image_url": "data/sample_images/chair_yellow.jpg",
      "description": "Iconic angled wooden frame lounge chair with vibrant mustard fabric.",
      "created_at": "2026-09-24T17:00:00Z",
      "similarity_score": 0.9642,
      "distance": 0.0358
    },
    {
      "id": 5,
      "name": "Solid Oak Dining Chair - Natural Finish",
      "category": "chair",
      "brand": "Nordic Living",
      "price": 160.0,
      "dimensions": "19W x 21D x 32H in",
      "material": "Solid White Oak",
      "color": "Natural Oak",
      "image_url": "data/sample_images/chair_oak.jpg",
      "description": "Sleek curved-back dining chair crafted from sustainable European oak.",
      "created_at": "2026-09-24T17:00:00Z",
      "similarity_score": 0.8120,
      "distance": 0.1880
    }
  ]
}
```

### 2. Check System Health
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "app_name": "Visual Product Matcher for Home-Design Catalogs",
  "database_connected": true,
  "catalog_items_count": 15,
  "embedding_dimension": 512,
  "vision_model": "resnet18"
}
```

---

## What I Learned

1. **Dual-Library Image Preprocessing**: Learned how combining Pillow (for robust byte decoding and EXIF correction) and OpenCV (for high-performance letterboxing and normalization) prevents common visual distortion bugs in computer vision pipelines.
2. **Feature Extraction Mechanics**: Learned how to strip classification heads (`fc = nn.Identity()`) from standard convolutional backbones to obtain compact, generalizable visual embeddings.
3. **Vector Database Integration**: Gained hands-on experience using PostgreSQL with pgvector, understanding the performance and operational trade-offs between HNSW and IVFFlat index types, and querying with the `<=>` cosine distance operator.
4. **API Safety & Error Handling**: Implemented boundary guards including file-type checking, upload size enforcement, and structured error responses.
5. **Effective Test Isolation**: Configured test fixtures using in-memory SQLite and mock models to ensure fast, deterministic testing without external dependencies.

---

## Future Improvements

- **Fine-Tuning on Furniture Datasets**: Fine-tune the backbone model using Triplet Loss or Contrastive Learning on domain-specific datasets (e.g., DeepFashion, Furniture-10k) to increase sensitivity to fine upholstery textures.
- **ONNX Runtime Acceleration**: Export the PyTorch feature extractor to ONNX format for reduced CPU inference latency.
- **Multi-Crop / Object Detection Pre-stage**: Add a lightweight bounding box detector (e.g., YOLOv8-nano) to isolate furniture pieces when users upload full room photos.
- **Caching**: Implement a Redis cache for frequent search queries and embeddings.

---

## Author

- **Harsha Raju**
- **GitHub**: [harsharaju1314-hash](https://github.com/harsharaju1314-hash)
- **Project Repository**: [Visual-Product-Matcher-for-Home-Design-Catalogs](https://github.com/harsharaju1314-hash/Visual-Product-Matcher-for-Home-Design-Catalogs)
