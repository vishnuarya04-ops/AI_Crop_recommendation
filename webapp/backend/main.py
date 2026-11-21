"""
FastAPI application — main entry point.

Routes:
    POST /api/predict     → Crop recommendation
    GET  /api/model-info  → Model metadata & performance
    GET  /api/health      → Health check
    GET  /docs            → Swagger UI (auto-generated)
"""

import time
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .config import APP_TITLE, APP_VERSION, APP_DESCRIPTION, CORS_ORIGINS, DEBUG
from .model import crop_model
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)


# ── Lifespan — load model on startup ───────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once when the server starts."""
    logger.info("🚀 Starting Crop Recommendation API v%s", APP_VERSION)
    try:
        crop_model.load()
        logger.info("✅ Server ready — model loaded successfully")
    except Exception as e:
        logger.error("❌ Failed to load model: %s", e)
        raise
    yield
    logger.info("👋 Shutting down")


# ── App ────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Logging Middleware ─────────────────────────
@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Log every request with timing and request ID."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    t0 = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    # Skip logging for static assets
    path = request.url.path
    if not path.startswith(("/css/", "/js/", "/favicon")):
        logger.info(
            "%s %s → %d (%.1fms) [%s]",
            request.method, path, response.status_code,
            elapsed_ms, request_id,
        )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
    return response


# ── Exception Handler ──────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — return structured JSON."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again.",
            "type": "internal_error",
        },
    )


# ════════════════════════════════════════════════════════
# API Routes
# ════════════════════════════════════════════════════════

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Get crop recommendation",
    tags=["Prediction"],
)
async def predict(req: PredictionRequest) -> PredictionResponse:
    """
    Accept soil nutrient levels and climate conditions,
    return the recommended crop with confidence scores.
    """
    result = crop_model.predict(
        n=req.N,
        p=req.P,
        k=req.K,
        temperature=req.temperature,
        humidity=req.humidity,
        ph=req.ph,
        rainfall=req.rainfall,
    )
    return PredictionResponse(**result)


@app.get(
    "/api/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata & performance metrics",
    tags=["Info"],
)
async def model_info() -> ModelInfoResponse:
    """Return model performance metrics, feature importance, and class list."""
    return ModelInfoResponse(**crop_model.get_info())


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Info"],
)
async def health() -> HealthResponse:
    """Check if the server and model are operational."""
    return HealthResponse(
        status="healthy",
        model_loaded=crop_model.is_loaded,
        model_name="Random Forest",
        version=APP_VERSION,
    )


# ════════════════════════════════════════════════════════
# Static files — serve the frontend
# ════════════════════════════════════════════════════════
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

# Serve index.html at root
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(_frontend_dir / "index.html")

@app.get("/about", include_in_schema=False)
async def about_page():
    return FileResponse(_frontend_dir / "about.html")

@app.get("/analysis", include_in_schema=False)
async def analysis_page():
    return FileResponse(_frontend_dir / "analysis.html")

# Favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(content={}, status_code=204)

# Mount static assets (css, js, images)
app.mount("/", StaticFiles(directory=str(_frontend_dir)), name="static")
