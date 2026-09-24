import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router
from app.db.session import init_db
from app.core.vision_model import get_vision_model

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Event:
    - Verifies database connection and tables.
    - Pre-warms the PyTorch vision model into memory before taking requests.
    """
    logger.info("Initializing Visual Product Matcher service...")
    init_db()
    # Pre-warm vision embedding model
    get_vision_model()
    logger.info("Vision model pre-warmed and ready.")
    yield
    logger.info("Visual Product Matcher service shutdown completed.")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="""
    **Visual Product Matcher for Home-Design Catalogs**
    
    A clean, realistic computer-vision REST API built with PyTorch, OpenCV, Pillow, FastAPI, and PostgreSQL with pgvector.
    Enables visual similarity search over furniture and home-decor catalogs.
    """,
    lifespan=lifespan
)

# CORS middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler for clean, interview-defensible JSON errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred while processing your visual catalog request."}
    )

# Include API routes under root or /api/v1
app.include_router(router)
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs_url": "/docs",
        "health_check": "/health",
        "search_endpoint": "/search"
    }
