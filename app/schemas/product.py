from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ProductBase(BaseModel):
    """Base schema for catalog product metadata."""
    name: str = Field(..., description="Product commercial name", json_schema_extra={"example": "Scandinavian Velvet Armchair"})
    category: str = Field(..., description="Furniture category (e.g. sofa, chair, table, cabinet, bed, lamp)", json_schema_extra={"example": "chair"})
    brand: Optional[str] = Field(None, description="Manufacturer or brand name", json_schema_extra={"example": "Nordic Living"})
    price: float = Field(..., gt=0, description="Retail price in USD", json_schema_extra={"example": 249.99})
    dimensions: Optional[str] = Field(None, description="Width x Depth x Height", json_schema_extra={"example": "32W x 34D x 36H in"})
    material: Optional[str] = Field(None, description="Primary material", json_schema_extra={"example": "Velvet & Solid Oak"})
    color: Optional[str] = Field(None, description="Product color shade", json_schema_extra={"example": "Mustard Yellow"})
    image_url: str = Field(..., description="URL or relative path to the catalog image", json_schema_extra={"example": "/images/armchair_yellow.jpg"})
    description: Optional[str] = Field(None, description="Detailed product description", json_schema_extra={"example": "Ergonomic accent chair with tapered wooden legs."})


class ProductCreate(ProductBase):
    """Schema for inserting a new product into the catalog."""
    pass


class ProductResponse(ProductBase):
    """Schema for returning a product record from PostgreSQL."""
    id: int
    created_at: Optional[datetime] = None

    model_config = {
        "from_attributes": True
    }


class ProductMatchResponse(ProductResponse):
    """Schema for a visual similarity search result with calculated score."""
    similarity_score: float = Field(..., description="Cosine similarity score between 0.0 and 1.0", json_schema_extra={"example": 0.915})
    distance: float = Field(..., description="Cosine distance (1.0 - similarity_score)", json_schema_extra={"example": 0.085})


class SearchResponse(BaseModel):
    """Schema for the visual search endpoint response."""
    total_results: int = Field(..., json_schema_extra={"example": 5})
    top_k: int = Field(..., json_schema_extra={"example": 5})
    query_image_info: Dict[str, Any] = Field(..., description="Image format, dimensions, and upload size")
    results: List[ProductMatchResponse] = Field(..., description="Ranked list of visually similar catalog items")


class HealthResponse(BaseModel):
    """Schema for the service health check endpoint."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    app_name: str
    database_connected: bool
    catalog_items_count: int
    embedding_dimension: int
    vision_model: str
