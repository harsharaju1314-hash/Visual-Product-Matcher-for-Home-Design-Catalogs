import numpy as np
import pytest
from app.db.models import Product


def test_product_model_attributes(db_session):
    """Verifies Product SQLAlchemy model field creation and querying."""
    product = db_session.query(Product).filter(Product.id == 1).first()
    assert product is not None
    assert product.name == "Test Modern Green Sofa"
    assert product.category == "sofa"
    assert float(product.price) == 799.00
    assert product.embedding is not None


def test_cosine_similarity_calculation():
    """
    Verifies cosine distance to cosine similarity mathematical conversion:
    cosine_sim = 1.0 - cosine_distance (for normalized unit vectors)
    """
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([1.0, 0.0, 0.0])
    vec_c = np.array([0.0, 1.0, 0.0])

    # Dot product similarity
    sim_identical = float(np.dot(vec_a, vec_b))
    sim_orthogonal = float(np.dot(vec_a, vec_c))

    assert sim_identical == 1.0
    assert sim_orthogonal == 0.0

    # Distance conversion
    dist_identical = 1.0 - sim_identical
    dist_orthogonal = 1.0 - sim_orthogonal

    assert dist_identical == 0.0
    assert dist_orthogonal == 1.0
