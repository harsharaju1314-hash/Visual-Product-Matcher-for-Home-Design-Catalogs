import numpy as np
import pytest
from app.core.preprocessor import ImagePreprocessor
from app.core.vision_model import VisionEmbeddingModel


def test_image_preprocessor_valid_jpeg(sample_valid_jpeg_bytes):
    """Verifies image loading, letterboxing, standardization, and shape."""
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    result = preprocessor.validate_and_preprocess(sample_valid_jpeg_bytes)

    assert result.original_format == "JPEG"
    assert result.original_size == (200, 200)
    assert result.file_size_bytes == len(sample_valid_jpeg_bytes)
    # Target batch input shape must be (1, 3, 224, 224)
    assert result.tensor_input.shape == (1, 3, 224, 224)
    assert result.tensor_input.dtype == np.float32


def test_image_preprocessor_rgba_conversion(sample_valid_png_bytes):
    """Verifies that RGBA images are safely converted to 3-channel RGB."""
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    result = preprocessor.validate_and_preprocess(sample_valid_png_bytes)

    assert result.original_format == "PNG"
    assert result.tensor_input.shape == (1, 3, 224, 224)


def test_image_preprocessor_corrupted_bytes():
    """Verifies that corrupt or non-image bytes raise a descriptive ValueError."""
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    corrupt_bytes = b"This is not a real image header or jpeg content."
    with pytest.raises(ValueError, match="Invalid or corrupted image"):
        preprocessor.validate_and_preprocess(corrupt_bytes)


def test_image_preprocessor_empty_bytes():
    """Verifies that empty byte inputs raise a ValueError."""
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    with pytest.raises(ValueError, match="empty"):
        preprocessor.validate_and_preprocess(b"")


def test_vision_model_embedding_extraction():
    """Verifies that PyTorch vision model extracts a 512-dim L2-normalized vector."""
    model = VisionEmbeddingModel(model_name="resnet18", embedding_dim=512)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    embedding = model.extract_embedding(dummy_input)

    assert isinstance(embedding, list)
    assert len(embedding) == 512
    # Verify L2-normalization: ||v||_2 == 1.0
    norm = np.linalg.norm(np.array(embedding))
    assert pytest.approx(norm, rel=1e-3) == 1.0
