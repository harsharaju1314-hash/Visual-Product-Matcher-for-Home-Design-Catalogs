from datetime import datetime
from sqlalchemy import Column, Integer, String, Numeric, Text, DateTime, func
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Product(Base):
    """
    SQLAlchemy Model representing a home-design furniture product.
    Includes relational metadata and a 512-dimensional embedding vector (pgvector).
    """
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    brand = Column(String(100), nullable=True)
    price = Column(Numeric(10, 2), nullable=False)
    dimensions = Column(String(100), nullable=True)
    material = Column(String(100), nullable=True)
    color = Column(String(50), nullable=True)
    image_url = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    
    # 512-dimensional vector column for ResNet-18 visual embeddings
    embedding = Column(Vector(512), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', category='{self.category}')>"
