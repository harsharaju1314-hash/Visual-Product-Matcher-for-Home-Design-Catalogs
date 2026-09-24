import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessingResult:
    """Container for preprocessed image array and original metadata."""
    tensor_input: np.ndarray       # Shape: (1, 3, 224, 224), float32 normalized
    original_format: str          # e.g., 'JPEG', 'PNG', 'WEBP'
    original_size: Tuple[int, int]# (width, height)
    file_size_bytes: int          # uploaded byte count


class ImagePreprocessor:
    """
    Dual-library image preprocessing pipeline:
    1. Pillow: Safe binary reading, format verification, and EXIF orientation handling.
    2. OpenCV: Aspect-ratio letterboxing/resizing, color space transformations, and channel-wise normalization.
    """

    IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_width, self.target_height = target_size

    def validate_and_preprocess(self, image_bytes: bytes) -> PreprocessingResult:
        """
        Validates raw image bytes and prepares them for PyTorch deep learning inference.
        """
        if not image_bytes:
            raise ValueError("Uploaded image file is empty (0 bytes).")

        file_size_bytes = len(image_bytes)

        # Step 1: Use Pillow for safe format verification and EXIF orientation correction
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()  # Verifies file integrity
            # Re-open after verify() as recommended by Pillow docs
            pil_img = Image.open(io.BytesIO(image_bytes))
            original_format = pil_img.format or "UNKNOWN"
            original_size = pil_img.size
            
            # Correct orientation from smartphone/camera EXIF data
            pil_img = ImageOps.exif_transpose(pil_img)
            
            # Convert to RGB (handles RGBA, Grayscale, CMYK, Palette formats safely)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
                
            # Convert PIL Image to NumPy array (RGB)
            rgb_array = np.array(pil_img)
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {str(e)}")

        # Step 2: Use OpenCV for aspect-ratio preserving resize (Letterboxing)
        letterboxed_rgb = self._letterbox_resize(rgb_array, self.target_width, self.target_height)

        # Step 3: Convert to float32 scaled to [0, 1]
        normalized = letterboxed_rgb.astype(np.float32) / 255.0

        # Step 4: Standardize using ImageNet Mean and Standard Deviation
        normalized = (normalized - self.IMAGE_NET_MEAN) / self.IMAGE_NET_STD

        # Step 5: Convert HWC (Height, Width, Channels) to CHW (Channels, Height, Width) for PyTorch
        # Resulting shape: (3, 224, 224)
        chw_array = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension: (1, 3, 224, 224)
        batched_tensor_input = np.expand_dims(chw_array, axis=0)

        return PreprocessingResult(
            tensor_input=batched_tensor_input,
            original_format=original_format,
            original_size=original_size,
            file_size_bytes=file_size_bytes
        )

    def _letterbox_resize(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Resizes an image using OpenCV while preserving its natural aspect ratio.
        Pads borders with neutral gray (128, 128, 128) to prevent geometric distortion.
        """
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # High-quality interpolation using OpenCV
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

        # Create target canvas filled with neutral gray
        canvas = np.full((target_h, target_w, 3), fill_value=128, dtype=np.uint8)

        # Calculate centering offsets
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2

        # Paste resized image into canvas center
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas
