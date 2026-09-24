import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import settings
from app.db.session import get_db
from app.db.models import Product
from app.schemas.product import (
    ProductCreate,
    ProductResponse,
    ProductMatchResponse,
    SearchResponse,
    HealthResponse
)
from app.core.preprocessor import ImagePreprocessor
from app.core.vision_model import VisionEmbeddingModel, get_vision_model

logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["Visual Catalog Search"])
preprocessor = ImagePreprocessor(target_size=(224, 224))


@router.get("/health", response_model=HealthResponse, summary="System Health & Status Check")
def health_check(
    db: Session = Depends(get_db),
    vision_model: VisionEmbeddingModel = Depends(get_vision_model)
):
    """
    Checks the status of the PostgreSQL database connection, pgvector extension,
    and vision model availability.
    """
    db_connected = False
    catalog_count = 0
    try:
        catalog_count = db.query(Product).count()
        db_connected = True
    except Exception as e:
        logger.error(f"Health check database query failed: {e}")

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        app_name=settings.APP_NAME,
        database_connected=db_connected,
        catalog_items_count=catalog_count,
        embedding_dimension=settings.EMBEDDING_DIM,
        vision_model=settings.MODEL_NAME
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Visual Similarity Search",
    description="Upload a furniture or room item image to retrieve visually similar products from the catalog."
)
async def search_similar_products(
    file: UploadFile = File(..., description="Query image file (JPEG, PNG, WEBP, max 5MB)"),
    top_k: int = Query(settings.DEFAULT_TOP_K, ge=1, le=20, description="Number of visual matches to return"),
    category: Optional[str] = Query(None, description="Optional category filter (e.g. sofa, chair, table, bed)"),
    db: Session = Depends(get_db),
    vision_model: VisionEmbeddingModel = Depends(get_vision_model)
):
    # 1. Validate file extension
    filename = file.filename or ""
    extension = filename.split(".")[-1].lower() if "." in filename else ""
    if extension not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Allowed formats: {settings.ALLOWED_IMAGE_EXTENSIONS}"
        )

    # 2. Read and validate upload size
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size ({len(contents)} bytes) exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_BYTES} bytes (5 MB)."
        )

    # 3. Preprocess image with Pillow & OpenCV
    try:
        prep_result = preprocessor.validate_and_preprocess(contents)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation error: {str(val_err)}"
        )
    except Exception as e:
        logger.error(f"Unexpected image preprocessing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query image."
        )

    # 4. Generate 512-dimensional embedding using PyTorch
    query_embedding = vision_model.extract_embedding(prep_result.tensor_input)

    # 5. Query PostgreSQL using pgvector cosine distance operator (<=>)
    try:
        distance_col = Product.embedding.cosine_distance(query_embedding).label("distance")
        query = db.query(Product, distance_col).filter(Product.embedding.isnot(None))

        if category and category.strip():
            query = query.filter(Product.category.ilike(category.strip()))

        matches = query.order_by("distance").limit(top_k).all()

        results: List[ProductMatchResponse] = []
        for product, distance in matches:
            dist_val = float(distance) if distance is not None else 1.0
            similarity = max(0.0, min(1.0, 1.0 - dist_val))

            match_item = ProductMatchResponse(
                id=product.id,
                name=product.name,
                category=product.category,
                brand=product.brand,
                price=float(product.price),
                dimensions=product.dimensions,
                material=product.material,
                color=product.color,
                image_url=product.image_url,
                description=product.description,
                created_at=product.created_at,
                similarity_score=round(similarity, 4),
                distance=round(dist_val, 4)
            )
            results.append(match_item)

        return SearchResponse(
            total_results=len(results),
            top_k=top_k,
            query_image_info={
                "format": prep_result.original_format,
                "dimensions": f"{prep_result.original_size[0]}x{prep_result.original_size[1]}",
                "file_size_bytes": prep_result.file_size_bytes
            },
            results=results
        )
    except Exception as db_err:
        logger.error(f"pgvector query error: {db_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector similarity search failed in database."
        )


@router.get(
    "/products/{product_id}",
    response_model=ProductResponse,
    summary="Retrieve Product by ID",
    description="Fetches detailed catalog metadata for a specific product ID."
)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found."
        )
    return product


@router.post(
    "/products",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New Catalog Product (Optional Admin Feature)",
    description="Registers product metadata in PostgreSQL. Embeddings can be generated during catalog ingestion."
)
def create_product(product_in: ProductCreate, db: Session = Depends(get_db)):
    db_product = Product(
        name=product_in.name,
        category=product_in.category.lower(),
        brand=product_in.brand,
        price=product_in.price,
        dimensions=product_in.dimensions,
        material=product_in.material,
        color=product_in.color,
        image_url=product_in.image_url,
        description=product_in.description
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product
