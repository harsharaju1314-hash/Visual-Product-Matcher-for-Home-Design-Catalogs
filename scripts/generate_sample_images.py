import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Directory for catalog and test images
IMG_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Define distinctive color palettes and geometry for furniture mock images
ITEMS = [
    ("sofa_green.jpg", (34, 139, 34), "SOFA - Forest Green"),
    ("sofa_gray.jpg", (169, 169, 169), "SOFA - Light Gray"),
    ("sofa_brown.jpg", (139, 69, 19), "SOFA - Cognac Brown"),
    ("chair_yellow.jpg", (218, 165, 32), "CHAIR - Mustard Yellow"),
    ("chair_oak.jpg", (205, 133, 63), "CHAIR - Natural Oak"),
    ("chair_stool.jpg", (50, 50, 50), "BAR STOOL - Matte Black"),
    ("table_walnut.jpg", (101, 67, 33), "TABLE - Dark Walnut"),
    ("table_marble.jpg", (245, 245, 240), "TABLE - White Marble"),
    ("table_desk.jpg", (210, 180, 140), "DESK - White Oak"),
    ("bed_charcoal.jpg", (70, 80, 90), "BED - Charcoal Gray"),
    ("bed_zen.jpg", (222, 184, 135), "BED - Zen Ash"),
    ("cabinet_teak.jpg", (160, 82, 45), "CABINET - Warm Teak"),
    ("cabinet_rattan.jpg", (210, 165, 110), "CABINET - Rattan Cane"),
    ("lamp_arc.jpg", (212, 175, 55), "LAMP - Brass Arc"),
    ("lamp_pendant.jpg", (30, 30, 30), "LAMP - Black Pendant"),
    ("query_yellow_chair.jpg", (218, 165, 32), "TEST QUERY - Yellow Chair"),
    ("query_green_sofa.jpg", (34, 139, 34), "TEST QUERY - Green Sofa")
]


def generate_images():
    """
    Generates clean 400x400 synthetic product images using OpenCV and Pillow
    to allow 100% offline and reproducible database seeding and API testing.
    """
    print(f"Generating synthetic furniture catalog images in: {IMG_DIR}")
    for filename, color_bgr, label in ITEMS:
        filepath = IMG_DIR / filename
        
        # Create base canvas (400x400)
        img = np.full((400, 400, 3), fill_value=245, dtype=np.uint8)
        
        # Draw background shadow rectangle
        cv2.rectangle(img, (45, 75), (355, 345), (220, 220, 220), -1)
        
        # Draw main furniture silhouette box with product color
        cv2.rectangle(img, (50, 80), (350, 340), color_bgr, -1)
        cv2.rectangle(img, (50, 80), (350, 340), (40, 40, 40), 2)
        
        # Draw decorative inner geometry (cushion / shelf lines)
        cv2.line(img, (50, 210), (350, 210), (30, 30, 30), 2)
        cv2.line(img, (200, 80), (200, 210), (30, 30, 30), 2)
        
        # Add text label using OpenCV
        cv2.putText(img, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
        
        # Save as high-quality JPEG
        cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  - Created {filename}")

    print("All sample images generated successfully.")


if __name__ == "__main__":
    generate_images()
