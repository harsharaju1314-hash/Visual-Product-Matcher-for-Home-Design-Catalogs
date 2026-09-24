import json
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import settings
from app.db.session import engine, SessionLocal, init_db
from app.db.models import Product
from app.core.preprocessor import ImagePreprocessor
from app.core.vision_model import get_vision_model
import scripts.generate_sample_images as sample_gen

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_catalog")


def seed_database():
    """
    Catalog Seeding Workflow:
    1. Initializes DB schema and pgvector extension.
    2. Generates local sample images if missing.
    3. Reads furniture catalog metadata from data/catalog_seed.json.
    4. Runs OpenCV/Pillow preprocessing & PyTorch ResNet-18 to compute 512-dim embeddings.
    5. Saves products and vector embeddings into PostgreSQL.
    """
    logger.info("Ensuring sample images exist...")
    sample_gen.generate_images()

    logger.info("Initializing database schema...")
    init_db()

    seed_file = Path(__file__).resolve().parent.parent / "data" / "catalog_seed.json"
    if not seed_file.exists():
        logger.error(f"Seed file not found: {seed_file}")
        return

    with open(seed_file, "r", encoding="utf-8") as f:
        catalog_items = json.load(f)

    preprocessor = ImagePreprocessor(target_size=(224, 224))
    vision_model = get_vision_model()
    project_root = Path(__file__).resolve().parent.parent

    db: Session = SessionLocal()
    try:
        # Check existing count
        existing_count = db.query(Product).count()
        if existing_count > 0:
            logger.info(f"Database already contains {existing_count} products. Clearing existing for fresh seed...")
            db.query(Product).delete()
            db.commit()

        logger.info(f"Seeding {len(catalog_items)} catalog products with PyTorch embeddings...")
        
        for item in catalog_items:
            img_path = project_root / item["image_url"]
            embedding = None

            if img_path.exists():
                try:
                    with open(img_path, "rb") as img_file:
                        raw_bytes = img_file.read()
                    prep_res = preprocessor.validate_and_preprocess(raw_bytes)
                    embedding = vision_model.extract_embedding(prep_res.tensor_input)
                except Exception as ex:
                    logger.warning(f"Could not compute embedding for {img_path}: {ex}")
            else:
                logger.warning(f"Image not found on disk: {img_path}")

            product = Product(
                name=item["name"],
                category=item["category"].lower(),
                brand=item.get("brand"),
                price=item["price"],
                dimensions=item.get("dimensions"),
                material=item.get("material"),
                color=item.get("color"),
                image_url=item["image_url"],
                description=item.get("description"),
                embedding=embedding
            )
            db.add(product)

        db.commit()
        total_seeded = db.query(Product).count()
        logger.info(f"Catalog seeding successfully completed! Total products: {total_seeded}")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during catalog seeding: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
