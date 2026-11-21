"""
API endpoint tests — uses FastAPI's TestClient (httpx-based).

Run:  cd webapp && python -m pytest tests/ -v
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure the webapp package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client that triggers model loading via lifespan."""
    with TestClient(app) as c:
        yield c


# ── Valid rice-like input ───────────────────────────────
VALID_INPUT = {
    "N": 80, "P": 48, "K": 40,
    "temperature": 23.5, "humidity": 82.0,
    "ph": 6.5, "rainfall": 236.0,
}

# ── Multiple crop scenarios ────────────────────────────
APPLE_INPUT = {
    "N": 20, "P": 130, "K": 200,
    "temperature": 22.6, "humidity": 92.0,
    "ph": 5.9, "rainfall": 113.0,
}

MAIZE_INPUT = {
    "N": 80, "P": 48, "K": 20,
    "temperature": 22.4, "humidity": 65.0,
    "ph": 6.2, "rainfall": 68.0,
}

COFFEE_INPUT = {
    "N": 100, "P": 18, "K": 30,
    "temperature": 25.5, "humidity": 58.0,
    "ph": 6.8, "rainfall": 158.0,
}


# ────────────────────────────────────────────────────────
# Health Tests
# ────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "Random Forest"

    def test_health_has_version(self, client):
        data = client.get("/api/health").json()
        assert "version" in data
        assert len(data["version"]) > 0

    def test_health_content_type(self, client):
        res = client.get("/api/health")
        assert "application/json" in res.headers["content-type"]


# ────────────────────────────────────────────────────────
# Prediction Tests
# ────────────────────────────────────────────────────────

class TestPredict:
    def test_valid_prediction(self, client):
        res = client.post("/api/predict", json=VALID_INPUT)
        assert res.status_code == 200
        data = res.json()
        assert "recommended_crop" in data
        assert "confidence" in data
        assert data["confidence"] > 0

    def test_response_structure(self, client):
        data = client.post("/api/predict", json=VALID_INPUT).json()
        # Top-level fields
        assert isinstance(data["recommended_crop"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["emoji"], str)
        assert isinstance(data["top3"], list)
        assert len(data["top3"]) == 3
        assert isinstance(data["all_probabilities"], dict)
        # Each top3 entry
        for crop_entry in data["top3"]:
            assert "crop" in crop_entry
            assert "probability" in crop_entry
            assert "emoji" in crop_entry

    def test_top3_crops_are_unique(self, client):
        """All top-3 recommended crops should be different."""
        data = client.post("/api/predict", json=VALID_INPUT).json()
        crops = [c["crop"] for c in data["top3"]]
        assert len(crops) == len(set(crops)), f"Duplicate crops in top-3: {crops}"

    def test_probabilities_sum_approximately_100(self, client):
        """All crop probabilities should sum to ~100%."""
        data = client.post("/api/predict", json=VALID_INPUT).json()
        total = sum(data["all_probabilities"].values())
        assert 99.0 <= total <= 101.0, f"Probabilities sum to {total}"

    def test_top1_matches_recommended(self, client):
        """The first entry in top3 should be the recommended crop."""
        data = client.post("/api/predict", json=VALID_INPUT).json()
        assert data["top3"][0]["crop"] == data["recommended_crop"]

    def test_confidence_matches_top1_probability(self, client):
        """Confidence should equal the top-1 probability."""
        data = client.post("/api/predict", json=VALID_INPUT).json()
        assert data["confidence"] == data["top3"][0]["probability"]

    def test_input_summary_echoed(self, client):
        """The response should echo back the input values."""
        data = client.post("/api/predict", json=VALID_INPUT).json()
        if data.get("input_summary"):
            assert data["input_summary"]["N"] == VALID_INPUT["N"]
            assert data["input_summary"]["ph"] == VALID_INPUT["ph"]

    def test_content_type_json(self, client):
        res = client.post("/api/predict", json=VALID_INPUT)
        assert "application/json" in res.headers["content-type"]

    def test_has_request_id_header(self, client):
        res = client.post("/api/predict", json=VALID_INPUT)
        assert "x-request-id" in res.headers

    def test_has_response_time_header(self, client):
        res = client.post("/api/predict", json=VALID_INPUT)
        assert "x-response-time" in res.headers

    # ── Multiple Scenario Tests ────────────────────────
    def test_apple_scenario(self, client):
        data = client.post("/api/predict", json=APPLE_INPUT).json()
        assert data["recommended_crop"] in [
            "apple", "grapes", "pomegranate"
        ], f"Expected apple-like crop, got {data['recommended_crop']}"
        assert data["confidence"] > 0

    def test_maize_scenario(self, client):
        data = client.post("/api/predict", json=MAIZE_INPUT).json()
        assert isinstance(data["recommended_crop"], str)
        assert data["confidence"] > 0

    def test_coffee_scenario(self, client):
        data = client.post("/api/predict", json=COFFEE_INPUT).json()
        assert isinstance(data["recommended_crop"], str)
        assert data["confidence"] > 0


# ────────────────────────────────────────────────────────
# Validation Tests
# ────────────────────────────────────────────────────────

class TestValidation:
    def test_invalid_ph_range(self, client):
        """pH = 15.0 is outside the valid range (3.5–10.0)."""
        bad_input = {**VALID_INPUT, "ph": 15.0}
        res = client.post("/api/predict", json=bad_input)
        assert res.status_code == 422

    def test_missing_field(self, client):
        """Omitting nitrogen should return a validation error."""
        incomplete = {k: v for k, v in VALID_INPUT.items() if k != "N"}
        res = client.post("/api/predict", json=incomplete)
        assert res.status_code == 422

    def test_negative_nitrogen(self, client):
        """N = -10 is below the minimum (0)."""
        bad_input = {**VALID_INPUT, "N": -10}
        res = client.post("/api/predict", json=bad_input)
        assert res.status_code == 422

    def test_temperature_too_high(self, client):
        """Temperature = 50 is above the maximum (44)."""
        bad_input = {**VALID_INPUT, "temperature": 50.0}
        res = client.post("/api/predict", json=bad_input)
        assert res.status_code == 422

    def test_humidity_too_low(self, client):
        """Humidity = 5 is below the minimum (14)."""
        bad_input = {**VALID_INPUT, "humidity": 5.0}
        res = client.post("/api/predict", json=bad_input)
        assert res.status_code == 422

    def test_rainfall_too_high(self, client):
        """Rainfall = 500 is above the maximum (300)."""
        bad_input = {**VALID_INPUT, "rainfall": 500.0}
        res = client.post("/api/predict", json=bad_input)
        assert res.status_code == 422

    def test_string_instead_of_number(self, client):
        """Passing a string for N should fail."""
        bad_input = {**VALID_INPUT, "N": "not_a_number"}
        res = client.post("/api/predict", json=bad_input)
        assert res.status_code == 422

    def test_empty_body(self, client):
        """Empty JSON body should fail."""
        res = client.post("/api/predict", json={})
        assert res.status_code == 422

    def test_error_response_format(self, client):
        """Validation errors should have proper structure."""
        bad_input = {**VALID_INPUT, "ph": 15.0}
        res = client.post("/api/predict", json=bad_input)
        data = res.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "msg" in data["detail"][0]


# ────────────────────────────────────────────────────────
# Edge Cases — Boundary Values
# ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_minimum_values(self, client):
        """All inputs at their minimum allowed values."""
        min_input = {
            "N": 0, "P": 5, "K": 5,
            "temperature": 8, "humidity": 14,
            "ph": 3.5, "rainfall": 20,
        }
        res = client.post("/api/predict", json=min_input)
        assert res.status_code == 200
        assert res.json()["confidence"] > 0

    def test_all_maximum_values(self, client):
        """All inputs at their maximum allowed values."""
        max_input = {
            "N": 140, "P": 145, "K": 205,
            "temperature": 44, "humidity": 100,
            "ph": 10.0, "rainfall": 300,
        }
        res = client.post("/api/predict", json=max_input)
        assert res.status_code == 200
        assert res.json()["confidence"] > 0

    def test_float_precision(self, client):
        """Inputs with high decimal precision should work."""
        precise_input = {
            "N": 80.12345, "P": 48.99999, "K": 40.00001,
            "temperature": 23.54321, "humidity": 82.11111,
            "ph": 6.543, "rainfall": 236.789,
        }
        res = client.post("/api/predict", json=precise_input)
        assert res.status_code == 200


# ────────────────────────────────────────────────────────
# Performance Tests
# ────────────────────────────────────────────────────────

class TestPerformance:
    def test_prediction_speed(self, client):
        """A single prediction should complete within 500ms."""
        t0 = time.perf_counter()
        res = client.post("/api/predict", json=VALID_INPUT)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert res.status_code == 200
        assert elapsed_ms < 500, f"Prediction took {elapsed_ms:.1f}ms (>500ms)"

    def test_health_check_speed(self, client):
        """Health check should respond within 100ms."""
        t0 = time.perf_counter()
        res = client.get("/api/health")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert res.status_code == 200
        assert elapsed_ms < 100, f"Health check took {elapsed_ms:.1f}ms (>100ms)"


# ────────────────────────────────────────────────────────
# Model Info Tests
# ────────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_returns_200(self, client):
        res = client.get("/api/model-info")
        assert res.status_code == 200

    def test_model_info_structure(self, client):
        data = client.get("/api/model-info").json()
        assert data["model_name"] == "Random Forest"
        assert data["test_accuracy"] == 99.55
        assert data["n_classes"] == 22
        assert len(data["classes"]) == 22
        assert len(data["features"]) == 7
        assert len(data["feature_importance"]) == 7

    def test_model_info_classes_are_sorted(self, client):
        """The class list should be consistent and complete."""
        data = client.get("/api/model-info").json()
        assert "rice" in data["classes"]
        assert "apple" in data["classes"]
        assert "coffee" in data["classes"]

    def test_features_list(self, client):
        data = client.get("/api/model-info").json()
        expected_features = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall"}
        assert set(data["features"]) == expected_features


# ────────────────────────────────────────────────────────
# Frontend Serving Tests
# ────────────────────────────────────────────────────────

class TestFrontend:
    def test_root_serves_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")

    def test_about_page_serves_html(self, client):
        res = client.get("/about")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")
