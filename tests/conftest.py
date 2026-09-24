import io
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import get_db
from app.db.models import Base, Product
from app.core.vision_model import VisionEmbeddingModel, get_vision_model

# In-memory SQLite database for unit and API testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


class MockVisionModel(VisionEmbeddingModel):
    """Deterministic mock vision model for fast, reproducible tests."""
    def __init__(self):
        self.embedding_dim = 512
        self.model_name = "resnet18_mock"

    def extract_embedding(self, preprocessed_numpy_input: np.ndarray):
        # Generate a deterministic normalized 512-dim embedding
        vec = np.ones(512, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist()


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Provides a clean database session for each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    # Seed with standard test products
    p1 = Product(
        id=1,
        name="Test Modern Green Sofa",
        category="sofa",
        brand="Test Brand",
        price=799.00,
        dimensions="80W x 34D in",
        material="Velvet",
        color="Green",
        image_url="data/sample_images/sofa_green.jpg",
        description="A comfortable green velvet sofa.",
        embedding=[0.04419] * 512  # Normalized 512-dim mock vector
    )
    p2 = Product(
        id=2,
        name="Test Yellow Accent Chair",
        category="chair",
        brand="Moderna",
        price=249.00,
        dimensions="30W x 30D in",
        material="Fabric",
        color="Yellow",
        image_url="data/sample_images/chair_yellow.jpg",
        description="Vibrant yellow accent armchair.",
        embedding=[0.04419] * 512
    )
    session.add(p1)
    session.add(p2)
    session.commit()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    """FastAPI TestClient with overridden DB and Vision Model dependencies."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def override_get_vision():
        return MockVisionModel()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_vision_model] = override_get_vision

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_valid_jpeg_bytes() -> bytes:
    """Generates a valid 200x200 RGB JPEG image in bytes."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_valid_png_bytes() -> bytes:
    """Generates a valid 200x200 RGBA PNG image in bytes."""
    img = Image.new("RGBA", (200, 200), color=(200, 100, 50, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
