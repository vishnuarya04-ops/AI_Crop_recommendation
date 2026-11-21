"""
Pydantic schemas — request/response models for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request ────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """Input features for crop recommendation."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "N": 80, "P": 48, "K": 40,
                    "temperature": 23.5, "humidity": 82.0,
                    "ph": 6.5, "rainfall": 236.0,
                }
            ]
        }
    }

    N: float = Field(
        ..., ge=0, le=140,
        description="Nitrogen content in soil (kg/ha)",
        examples=[80],
    )
    P: float = Field(
        ..., ge=5, le=145,
        description="Phosphorus content in soil (kg/ha)",
        examples=[48],
    )
    K: float = Field(
        ..., ge=5, le=205,
        description="Potassium content in soil (kg/ha)",
        examples=[40],
    )
    temperature: float = Field(
        ..., ge=8, le=44,
        description="Average temperature during growing season (°C)",
        examples=[23.5],
    )
    humidity: float = Field(
        ..., ge=14, le=100,
        description="Relative humidity (%)",
        examples=[82.0],
    )
    ph: float = Field(
        ..., ge=3.5, le=10.0,
        description="Soil pH value",
        examples=[6.5],
    )
    rainfall: float = Field(
        ..., ge=20, le=300,
        description="Annual rainfall (mm)",
        examples=[236.0],
    )


# ── Response models ────────────────────────────────────
class CropPrediction(BaseModel):
    """Single crop prediction with metadata."""
    crop: str
    probability: float
    emoji: str
    info: str


class InputSummary(BaseModel):
    """Echo of the input values."""
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# ── Soil Analysis Models (NEW) ─────────────────────────
class MoistureAnalysis(BaseModel):
    """Soil moisture level estimation."""
    level: str
    percentage: float
    description: str


class MineralAssessment(BaseModel):
    """Assessment of a single mineral/feature."""
    value: float
    status: str
    color: str
    unit: str
    name: str


class FertilityBreakdown(BaseModel):
    """Breakdown of fertility score components."""
    npk_score: float
    balance_score: float
    ph_score: float
    moisture_score: float


class FertilityScore(BaseModel):
    """Composite soil fertility score."""
    score: float
    grade: str
    description: str
    breakdown: FertilityBreakdown


class FeatureMatch(BaseModel):
    """Per-feature suitability match for a crop."""
    feature: str
    name: str
    unit: str
    your_value: float
    ideal_min: float
    ideal_max: float
    tolerable_min: float
    tolerable_max: float
    score: float
    status: str
    color: str


class CropSuitability(BaseModel):
    """Suitability analysis for a single crop."""
    crop: str
    emoji: Optional[str] = "🌱"
    overall_match: float
    verdict: str
    verdict_text: Optional[str] = ""
    should_plant: Optional[bool] = False
    features: list[FeatureMatch]
    tips: list[str]


class SoilAnalysis(BaseModel):
    """Complete soil analysis result."""
    moisture: MoistureAnalysis
    minerals: dict[str, MineralAssessment]
    fertility: FertilityScore
    crop_suitability: list[CropSuitability]


# ── Main Prediction Response ───────────────────────────
class PredictionResponse(BaseModel):
    """Full prediction result returned by /api/predict."""
    recommended_crop: str
    confidence: float
    emoji: str
    info: str
    top3: list[CropPrediction]
    all_probabilities: dict[str, float]
    input_summary: Optional[InputSummary] = None
    soil_analysis: Optional[SoilAnalysis] = None


class FeatureImportance(BaseModel):
    rank: int
    feature: str
    avg_rank: float


class ModelInfoResponse(BaseModel):
    """Model metadata returned by /api/model-info."""
    model_name: str
    n_estimators: Optional[int] = None
    test_accuracy: float
    train_accuracy: float
    f1_macro: float
    cv_accuracy: float
    cv_std: float
    n_classes: int
    classes: list[str]
    features: list[str]
    feature_importance: list[FeatureImportance]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    version: str
