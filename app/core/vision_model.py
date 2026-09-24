import logging
from typing import List
import numpy as np
import torch
import torch.nn as nn
from app.config import settings

logger = logging.getLogger("uvicorn.error")


class VisionEmbeddingModel:
    """
    PyTorch Feature Extractor using a pretrained ResNet-18 backbone.
    Strips the final classification layer to produce dense 512-dimensional
    L2-normalized visual embeddings.
    """

    def __init__(self, model_name: str = "resnet18", embedding_dim: int = 512):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """
        Loads pretrained ResNet-18 weights and replaces the FC classification layer with Identity.
        Falls back to a feature extractor module if torchvision downloads are restricted.
        """
        try:
            import torchvision.models as models
            try:
                # Modern torchvision weights enum syntax
                from torchvision.models import ResNet18_Weights
                base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                base_model = models.resnet18(pretrained=True)

            # Strip the 1000-class classifier head; keep only the 512-dim feature representation
            base_model.fc = nn.Identity()
            base_model.eval()
            base_model.to(self.device)
            logger.info(f"Loaded pretrained {self.model_name} feature extractor on {self.device}.")
            return base_model
        except Exception as e:
            logger.warning(f"Could not load torchvision model directly ({e}). Initializing standalone Torch ResNet architecture.")
            # Fallback lightweight CNN feature extractor for isolated testing
            class LightweightFeatureExtractor(nn.Module):
                def __init__(self, out_dim=512):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(64, out_dim)
                    )

                def forward(self, x):
                    return self.features(x)

            fallback = LightweightFeatureExtractor(self.embedding_dim)
            fallback.eval()
            fallback.to(self.device)
            return fallback

    def extract_embedding(self, preprocessed_numpy_input: np.ndarray) -> List[float]:
        """
        Performs forward inference with torch.no_grad() and L2-normalizes the vector.
        Input shape: (1, 3, 224, 224)
        Output: List[float] of length 512 where sum(x_i^2) == 1.0
        """
        tensor_input = torch.from_numpy(preprocessed_numpy_input).to(self.device)

        with torch.no_grad():
            features = self.model(tensor_input)
            # Flatten to 1D vector: shape (512,)
            vector = features.squeeze(0).cpu().numpy()

            # L2-normalization: unit vector enables cosine similarity via dot product
            norm = np.linalg.norm(vector)
            if norm > 0:
                normalized_vector = vector / norm
            else:
                normalized_vector = vector

        return normalized_vector.tolist()


# Global Singleton instance for fast API request lifecycle
_vision_model_instance: VisionEmbeddingModel = None


def get_vision_model() -> VisionEmbeddingModel:
    """Returns the singleton vision embedding model instance."""
    global _vision_model_instance
    if _vision_model_instance is None:
        _vision_model_instance = VisionEmbeddingModel(
            model_name=settings.MODEL_NAME,
            embedding_dim=settings.EMBEDDING_DIM
        )
    return _vision_model_instance
