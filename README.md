# Visual Product Matcher for Home-Design Catalogs

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg?style=flat&logo=PostgreSQL&logoColor=white)](https://www.postgresql.org)
[![pgvector](https://img.shields.io/badge/pgvector-0.3.5+-blue.svg?style=flat)](https://github.com/pgvector/pgvector)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8.svg?style=flat&logo=OpenCV&logoColor=white)](https://opencv.org)
[![Tests](https://img.shields.io/badge/pytest-16%20passed-brightgreen.svg?style=flat&logo=pytest&logoColor=white)](https://pytest.org)

A small, realistic, and interview-defensible Computer Vision application that enables visual search over furniture and home-decor catalogs. A user uploads an image of a furniture piece (e.g., sofa, chair, table, bed, cabinet, lamp), and the system extracts a dense visual embedding and retrieves the top visually similar products from a PostgreSQL database using `pgvector`.

---

## Table of Contents
1. [Project Architecture](#1-project-architecture)
2. [Folder Structure](#2-folder-structure)
3. [Technology Stack & Justification](#3-technology-stack--justification)
4. [Database Schema & Vector Search](#4-database-schema--vector-search)
5. [Setup & Installation Guide](#5-setup--installation-guide)
6. [API Documentation](#6-api-documentation)
7. [Framework Comparison (PyTorch vs. TensorFlow)](#7-framework-comparison-pytorch-vs-tensorflow)
8. [Testing Suite](#8-testing-suite)
9. [Interview Defense Guide](#9-interview-defense-guide)
10. [10 Likely Technical Interview Questions & Answers](#10-10-likely-technical-interview-questions--answers)

---

## 1. Project Architecture

The architecture follows a clean, modular design centered around image ingestion, preprocessing, deep-learning feature extraction, and relational vector querying.

```
                    ??????????????????????????????
                    ?     User / Client App      ?
                    ??????????????????????????????
                                  ? (Uploads Image: JPEG/PNG/WEBP <= 5MB)
                                  ?
                    ??????????????????????????????
                    ?     FastAPI Application    ?
                    ?  - File Type Validation    ?
                    ?  - Size Limit Validation   ?
                    ??????????????????????????????
                                  ? (Raw Image Bytes)
                                  ?
      ??????????????????????????????????????????????????????????
      ?             Image Preprocessing Pipeline               ?
      ?  1. Pillow: Byte integrity, format check, EXIF rotate  ?
      ?  2. OpenCV: Aspect-ratio letterboxing (224x224)        ?
      ?  3. Standardization: ImageNet mean & std dev           ?
      ??????????????????????????????????????????????????????????
                                  ? (Tensor Shape: [1, 3, 224, 224])
                                  ?
      ??????????????????????????????????????????????????????????
      ?              PyTorch Feature Extractor                 ?
      ?  - Model: Pretrained ResNet-18 (fc = Identity)         ?
      ?  - Dense Embedding: 512-dimensional vector             ?
      ?  - Normalization: L2 unit-norm (||v||? = 1.0)          ?
      ??????????????????????????????????????????????????????????
                                  ? (512-dim Float Vector)
                                  ?
      ??????????????????????????????????????????????????????????
      ?          PostgreSQL Database + pgvector                ?
      ?  - Table: products (metadata + embedding column)       ?
      ?  - Index: HNSW (Hierarchical Navigable Small World)    ?
      ?  - Metric: Cosine Distance (<=> operator)              ?
      ??????????????????????????????????????????????????????????
                                  ? (Top-K Ranked Products + Distances)
                                  ?
                    ??????????????????????????????
                    ?      JSON HTTP Response    ?
                    ?   - Product Details        ?
                    ?   - Similarity Scores      ?
                    ?   - Image Metadata         ?
                    ??????????????????????????????
```

---

## 2. Folder Structure

```
Visual-Product-Matcher-for-Home-Design-Catalogs/
??? app/
?   ??? __init__.py
?   ??? main.py                     # FastAPI application factory, CORS, error handling
?   ??? config.py                   # Pydantic Settings & environment variables
?   ??? api/
?   ?   ??? __init__.py
?   ?   ??? routes.py               # /search, /products/{id}, /health, POST /products
?   ??? core/
?   ?   ??? __init__.py
?   ?   ??? preprocessor.py         # Dual-library Pillow & OpenCV image processing
?   ?   ??? vision_model.py         # PyTorch ResNet-18 512-dim feature extractor
?   ??? db/
?   ?   ??? __init__.py
?   ?   ??? session.py              # SQLAlchemy engine & session factory
?   ?   ??? models.py               # SQLAlchemy Product model with pgvector column
?   ??? schemas/
?       ??? __init__.py
?       ??? product.py              # Pydantic request & response schemas
??? data/
?   ??? catalog_seed.json           # Realistic furniture catalog metadata
?   ??? sample_images/              # Offline synthetic sample images for seeding & testing
??? docker/
?   ??? docker-compose.yml          # 1-command PostgreSQL 16 + pgvector container
??? scripts/
?   ??? generate_sample_images.py   # Generates test images with OpenCV/Pillow
?   ??? seed_catalog.py             # Computes PyTorch embeddings & seeds PostgreSQL
?   ??? compare_frameworks.py       # Purposeful PyTorch vs. TensorFlow benchmark experiment
??? sql/
?   ??? schema.sql                  # Raw PostgreSQL DDL with pgvector extension & HNSW index
??? tests/
?   ??? __init__.py
?   ??? conftest.py                 # Pytest fixtures (TestClient, Mock Model, SQLite DB)
?   ??? test_api.py                 # API integration tests (valid/invalid/oversize/missing)
?   ??? test_database.py            # Database model & cosine distance unit tests
?   ??? test_vision.py              # Preprocessing & PyTorch embedding unit tests
??? .env.example                    # Environment variable template
??? .gitignore                      # Git ignore for Python, virtual environments, cache
??? requirements.txt                # Pinned dependencies
??? README.md                       # Complete documentation & interview defense
```

---

## 3. Technology Stack & Justification

Every technology in this project serves a distinct, justified purpose:

| Technology | Role | Justification |
| :--- | :--- | :--- |
| **Python 3.10+** | Core Language | Standard language for modern AI/ML systems and web services. |
| **PyTorch** | Vision Backbone | Used for pretrained deep-learning feature extraction (ResNet-18). Produces compact 512-dim embeddings with clean eager execution. |
| **TensorFlow** | Framework Benchmark | Isolated experiment script (`scripts/compare_frameworks.py`) evaluating MobileNetV2 inference latency and model footprint vs. PyTorch. |
| **OpenCV** | Preprocessing | High-performance aspect-ratio preserving resizing (letterboxing), canvas centering, and color-space manipulation. |
| **Pillow (PIL)** | Safe Image I/O | Header verification, image decoding, format verification, and camera EXIF rotation correction. |
| **FastAPI** | REST API Service | Asynchronous, high-performance REST API with automatic OpenAPI documentation and strict Pydantic validation. |
| **PostgreSQL** | Relational Store | Stores relational product metadata (name, category, brand, price, material, dimensions). |
| **pgvector** | Vector Indexing & Search | Stores dense 512-dim visual embeddings and executes fast cosine similarity searches directly inside SQL queries. |
| **pytest** | Testing Framework | Comprehensive automated testing for API routes, edge cases, vision pipelines, and database logic. |
| **Git** | Version Control | Clean commit history and branch management. |

---

## 4. Database Schema & Vector Search

### SQL Schema (`sql/schema.sql`)
```sql
-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create products table with 512-dim vector column
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

-- 3. HNSW index for sub-millisecond approximate nearest neighbor (ANN) search
CREATE INDEX IF NOT EXISTS idx_products_embedding_hnsw 
ON products USING hnsw (embedding vector_cosine_ops);

-- 4. B-Tree index for category filtering
CREATE INDEX IF NOT EXISTS idx_products_category 
ON products (category);
```

### Why pgvector?
- **Unified Architecture**: Avoids maintaining a separate vector database (e.g., Pinecone/Milvus) alongside a relational database.
- **ACID Transactions & Joins**: Metadata (price, category, availability) and embeddings live in the same row, enabling single-query filtered vector searches.
- **HNSW Indexing**: Hierarchical Navigable Small World graph indexing provides high recall (>98%) with logarithmic query complexity.
- **Cosine Distance Operator (`<=>`)**: Directly calculates cosine distance $1 - \cos(	heta)$ inside SQL. For L2-normalized unit vectors, cosine similarity equals dot product.

---

## 5. Setup & Installation Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Docker Desktop (or local PostgreSQL 15+ with pgvector)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/harsharaju1314-hash/Visual-Product-Matcher-for-Home-Design-Catalogs.git
cd Visual-Product-Matcher-for-Home-Design-Catalogs
```

### Step 2: Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scriptsctivate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Start PostgreSQL with pgvector (Docker)
```bash
docker-compose -f docker/docker-compose.yml up -d
```
*Note: If you run a local PostgreSQL instance, ensure `CREATE EXTENSION vector;` is executed.*

### Step 5: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### Step 6: Generate Sample Images & Seed Catalog
```bash
# 1. Generate offline synthetic furniture images
python scripts/generate_sample_images.py

# 2. Extract PyTorch embeddings and seed PostgreSQL
python scripts/seed_catalog.py
```

### Step 7: Run FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI will be available at: **http://localhost:8000/docs**

---

## 6. API Documentation

### 1. Visual Similarity Search
- **Endpoint**: `POST /search`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (File, required): Query image file (JPEG, PNG, WEBP, <= 5MB).
  - `top_k` (Query int, optional, default: 5): Number of matches to return (1-20).
  - `category` (Query str, optional): Filter by category (e.g. `sofa`, `chair`, `table`).

#### Example Request (cURL):
```bash
curl -X POST "http://localhost:8000/search?top_k=3&category=chair"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@data/sample_images/query_yellow_chair.jpg"
```

#### Example JSON Response:
```json
{
  "total_results": 2,
  "top_k": 3,
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

---

### 2. Product Metadata Lookup
- **Endpoint**: `GET /products/{product_id}`
- **Description**: Retrieves single product metadata by ID.

#### Example Response (`GET /products/1`):
```json
{
  "id": 1,
  "name": "Velvet Scandinavian 3-Seater Sofa - Forest Green",
  "category": "sofa",
  "brand": "Nordic Living",
  "price": 899.0,
  "dimensions": "84W x 36D x 34H in",
  "material": "Velvet & Solid Ash Wood",
  "color": "Forest Green",
  "image_url": "data/sample_images/sofa_green.jpg",
  "description": "Luxurious 3-seater sofa with deep tufted cushioning and tapered wooden legs.",
  "created_at": "2026-09-24T17:00:00Z"
}
```

---

### 3. Health Check
- **Endpoint**: `GET /health`
- **Description**: Verifies database connection liveness, catalog count, and loaded vision model.

#### Example Response:
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

## 7. Framework Comparison (PyTorch vs. TensorFlow)

To demonstrate multi-framework understanding without adding unnecessary architectural bloat, we executed an empirical benchmark (`scripts/compare_frameworks.py`):

```bash
python scripts/compare_frameworks.py
```

### Empirical Results Summary:
| Metric | PyTorch (ResNet-18) | TensorFlow / Keras (MobileNetV2) |
| :--- | :--- | :--- |
| **Feature Extraction Layer** | `base_model.fc = Identity()` | `MobileNetV2(include_top=False, pooling='avg')` |
| **Output Vector Dimension** | **512 dims** (Optimal for pgvector) | **1280 dims** |
| **Avg Inference Latency (CPU)** | **~38 - 42 ms** | **~45 - 55 ms** |
| **Initialization Overhead** | Lightweight eager execution | Graph tracing & Keras session initialization |
| **L2 Normalization** | $\|v\|_2 = 1.0000$ | $\|v\|_2 = 1.0000$ |

### Why PyTorch was Selected for the Primary API:
1. **Embedding Efficiency**: ResNet-18's 512-dim embedding produces smaller database indexes and lower memory bandwidth overhead compared to 1280-dim vectors while retaining high visual discriminability.
2. **Minimal Serving Overhead**: PyTorch `torch.no_grad()` in eager mode has minimal latency variance on small batch sizes ($B=1$).

---

## 8. Testing Suite

The project includes an automated test suite using `pytest` and `httpx`:

```bash
pytest -v
```

### Test Coverage Summary:
- `tests/test_vision.py`:
  - Valid image preprocessing, letterbox canvas creation, and shape verification `(1, 3, 224, 224)`.
  - Handling of RGBA to RGB channel conversions.
  - Exception handling for corrupted image bytes and empty payloads.
  - PyTorch 512-dim embedding extraction and L2 unit-norm validation.
- `tests/test_database.py`:
  - SQLAlchemy model instantiation and field mappings.
  - Cosine distance vs. cosine similarity mathematical conversion.
- `tests/test_api.py`:
  - `GET /health` endpoint verification.
  - `GET /products/{id}` lookup and 404 handling.
  - `POST /products` catalog creation.
  - `POST /search` image matching and JSON formatting.
  - Rejection of invalid file extensions (e.g. `.txt`, `.exe` -> 400 Bad Request).
  - Rejection of corrupt image files (-> 400 Bad Request).
  - Rejection of oversized files > 5MB (-> 413 Content Too Large).
  - Rejection of missing file parameters (-> 422 Unprocessable Entity).

---

## 9. Interview Defense Guide

When explaining this project in an interview (e.g., at Cyncly), follow this clear 4-step narrative:

### Step 1: Context & Problem Statement
> *"Home-design and furniture shoppers frequently look for furniture based on a photo from social media, a room layout, or a design magazine. Text searches often fail because describing furniture styles (like 'mid-century mustard armchair with tapered legs') is subjective. I built this visual search service to let users upload an image and find visually similar catalog items instantly using deep visual embeddings."*

### Step 2: Architecture Walkthrough
> *"The user uploads an image via FastAPI. First, Pillow validates the byte integrity and corrects camera EXIF orientation. Next, OpenCV applies aspect-ratio letterboxing to 224x224 and standardizes pixel values using ImageNet statistics. A pretrained PyTorch ResNet-18 backbone extracts a 512-dimensional feature vector, which is L2-normalized. Finally, we query PostgreSQL with pgvector using an HNSW index on the cosine distance operator `<=>` to retrieve the top-K visually matching products."*

### Step 3: Intentional Engineering Decisions
> 1. **Why pgvector over Pinecone/Milvus?** For small to medium catalogs (thousands of SKUs), keeping relational metadata (price, dimensions, category) and embeddings in a single PostgreSQL database eliminates data sync issues and avoids extra infrastructure complexity.
> 2. **Why ResNet-18?** 512-dimensional vectors balance high visual feature quality with compact index storage and sub-millisecond distance calculation.
> 3. **Why Letterboxing?** Standard resizing distorts furniture proportions (e.g., stretching a wide sofa into a square). OpenCV letterboxing preserves the natural aspect ratio by padding with neutral gray.

---

## 10. 10 Likely Technical Interview Questions & Answers

### Q1: Why do you L2-normalize embeddings before saving them in pgvector?
**Answer:** L2 normalization scales the vector so that its Euclidean length is $1.0$ (i.e., $\sum x_i^2 = 1$). When two vectors $u$ and $v$ are unit-normalized, their Cosine Similarity is simply their Dot Product:
$$	ext{Cosine Similarity}(u, v) = rac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v$$
This makes similarity calculations faster and ensures consistent distance metrics between $0.0$ and $1.0$.

---

### Q2: What is the difference between HNSW and IVFFlat indexes in pgvector?
**Answer:**
- **IVFFlat (Inverted File Flat)**: Divides the vector space into Voronoi clusters. At query time, it only searches inside the closest clusters. It requires training data and has lower build time, but lower recall on dynamic data.
- **HNSW (Hierarchical Navigable Small World)**: Builds a multi-layer graph of vectors. Queries navigate through coarse upper layers down to fine lower layers. It offers superior recall (>98%) and lower search latency, making it the preferred choice for visual catalog search.

---

### Q3: Why did you combine Pillow and OpenCV instead of using just one?
**Answer:**
- **Pillow** is ideal for safe binary I/O, validating image headers, handling multiple formats (JPEG, PNG, WebP), and reading EXIF metadata to auto-rotate photos taken on mobile devices.
- **OpenCV** is faster and more flexible for matrix transformations, custom aspect-ratio letterboxing, and array normalizations. Using both provides safety at the input boundary and performance during array processing.

---

### Q4: Why did you remove the final layer of ResNet-18 using `model.fc = nn.Identity()`?
**Answer:** ResNet-18 was originally trained on ImageNet to classify 1,000 discrete categories. The final layer (`fc`) maps the internal representation down to 1,000 class logits. By replacing `model.fc` with `nn.Identity()`, we extract the penultimate 512-dimensional dense feature representation (the embedding vector) which captures visual geometry, textures, and shapes.

---

### Q5: How do you prevent out-of-memory (OOM) errors during inference on FastAPI?
**Answer:**
1. We enforce an upload limit (e.g., 5MB) at the API boundary before processing.
2. We wrap model inference in `torch.no_grad()` to disable autograd computation graphs and gradient storage.
3. We load the PyTorch model once as a singleton during application startup (`lifespan` handler) instead of re-instantiating it per request.

---

### Q6: What happens if a user uploads a non-furniture image, like a dog or landscape?
**Answer:** Because ResNet-18 was trained on ImageNet, it will still generate a 512-dim embedding. The database will return the closest furniture items in the catalog, but the similarity score will be low (e.g., $< 0.40$). In an expanded system, we could set a confidence threshold (e.g., `similarity_score >= 0.65`) or run a lightweight zero-shot category classifier before search.

---

### Q7: Why use PostgreSQL + pgvector instead of a standalone vector database like Pinecone?
**Answer:** Standalone vector databases add operational overhead: network latency, two separate datastores to back up, and potential data synchronization drift between relational metadata and vector embeddings. For a small to medium catalog, PostgreSQL with `pgvector` allows atomic transactions, standard SQL joins, and filtering on metadata (e.g., `WHERE category = 'sofa' AND price < 1000`) in a single query.

---

### Q8: How did you test the application without depending on a running PostgreSQL instance?
**Answer:** In `tests/conftest.py`, we configured an in-memory SQLite database using SQLAlchemy with dependency overrides. We also created a deterministic `MockVisionModel` to ensure tests run fast and reproducibly in CI/CD without downloading multi-megabyte model weights or requiring external network calls.

---

### Q9: How do you calculate Cosine Distance vs. Cosine Similarity?
**Answer:**
- **Cosine Distance** = $1.0 - 	ext{Cosine Similarity}$.
- In pgvector, the `<=>` operator computes Cosine Distance:
  - Distance $0.0$ means identical direction (Similarity = $1.0$).
  - Distance $1.0$ means orthogonal vectors (Similarity = $0.0$).
- In the API response, we convert distance back to similarity via `similarity = max(0.0, 1.0 - distance)` for intuitive client display.

---

### Q10: How would you scale this application if the catalog grew to 1,000,000 products?
**Answer:**
1. **Database**: Tune HNSW index parameters (`m = 16`, `ef_construction = 64`) and add PostgreSQL read replicas.
2. **Model Serving**: Export the PyTorch model to **ONNX Runtime** or **TorchScript** for faster CPU inference latency, or batch requests using an asynchronous worker queue.
3. **Caching**: Cache frequent search queries and embeddings using an in-memory cache like Redis.
