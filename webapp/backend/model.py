"""
Model layer — loads the trained Random Forest and preprocessing pipeline,
exposes a clean predict() interface with caching and logging.

Also includes the SoilAnalyzer class for detailed soil analysis,
mineral profiling, crop suitability matching, and planting advisories.
"""

import time
import hashlib
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from .config import MODEL_PATH, PIPELINE_PATH

logger = logging.getLogger(__name__)


# ── Crop metadata ───────────────────────────────────────
CROP_EMOJIS: dict[str, str] = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘",
    "chickpea": "🫛", "coconut": "🥥", "coffee": "☕",
    "cotton": "🧵", "grapes": "🍇", "jute": "🪢",
    "kidneybeans": "🫘", "lentil": "🍲", "maize": "🌽",
    "mango": "🥭", "mothbeans": "🫘", "mungbean": "🫛",
    "muskmelon": "🍈", "orange": "🍊", "papaya": "🍈",
    "pigeonpeas": "🫛", "pomegranate": "🫐", "rice": "🍚",
    "watermelon": "🍉",
}

CROP_INFO: dict[str, str] = {
    "rice":        "Thrives in warm, humid climates with heavy rainfall.",
    "maize":       "Grows well in moderate climates with well-drained soil.",
    "chickpea":    "Prefers cool, dry conditions with light to moderate rainfall.",
    "kidneybeans": "Needs moderate temperature and rainfall with neutral pH soil.",
    "pigeonpeas":  "Tolerates semi-arid conditions and performs well in poor soils.",
    "mothbeans":   "Drought-resistant legume suited for arid and semi-arid regions.",
    "mungbean":    "Grows best in warm climates with well-distributed rainfall.",
    "blackgram":   "Prefers warm, humid conditions with moderate rainfall.",
    "lentil":      "Thrives in cool seasons with low to moderate rainfall.",
    "pomegranate": "Prefers semi-arid conditions with hot, dry summers.",
    "banana":      "Requires tropical warmth, high humidity, and ample water.",
    "mango":       "Grows best in tropical and subtropical climates.",
    "grapes":      "Prefers warm, dry summers with well-drained soil.",
    "watermelon":  "Needs warm temperatures and long growing seasons.",
    "muskmelon":   "Thrives in hot, dry climates with sandy loam soil.",
    "apple":       "Requires cold winters and moderate summers.",
    "orange":      "Grows well in subtropical climates with moderate rainfall.",
    "papaya":      "Needs tropical warmth and consistent moisture.",
    "coconut":     "Thrives in coastal tropical regions with sandy soil.",
    "cotton":      "Grows best in warm climates with black soil.",
    "jute":        "Requires hot, humid conditions with heavy rainfall.",
    "coffee":      "Prefers tropical highlands with moderate temperature.",
}

# ── Ideal crop ranges (computed from training dataset) ──
# Each crop has Q25–Q75 (ideal range) and min–max (tolerable range)
# from the Crop_recommendation.csv training data.
CROP_IDEAL_RANGES: dict[str, dict] = {
    "apple": {
        "N": {"min": 0.0, "max": 40.0, "mean": 20.8, "q25": 10.0, "q75": 32.0},
        "P": {"min": 125.0, "max": 145.0, "mean": 134.22, "q25": 129.0, "q75": 139.0},
        "K": {"min": 195.0, "max": 205.0, "mean": 199.89, "q25": 197.0, "q75": 203.0},
        "temperature": {"min": 21.0, "max": 23.99, "mean": 22.63, "q25": 21.82, "q75": 23.47},
        "humidity": {"min": 90.04, "max": 94.84, "mean": 92.33, "q25": 91.02, "q75": 93.56},
        "ph": {"min": 5.5, "max": 6.5, "mean": 5.93, "q25": 5.68, "q75": 6.15},
        "rainfall": {"min": 100.17, "max": 134.55, "mean": 112.65, "q25": 105.09, "q75": 118.99},
    },
    "banana": {
        "N": {"min": 80.0, "max": 120.0, "mean": 100.23, "q25": 90.0, "q75": 110.0},
        "P": {"min": 70.0, "max": 95.0, "mean": 82.01, "q25": 75.75, "q75": 89.0},
        "K": {"min": 45.0, "max": 55.0, "mean": 50.05, "q25": 47.0, "q75": 53.0},
        "temperature": {"min": 25.01, "max": 29.99, "mean": 27.38, "q25": 26.1, "q75": 28.59},
        "humidity": {"min": 75.05, "max": 84.84, "mean": 80.36, "q25": 77.5, "q75": 82.87},
        "ph": {"min": 5.5, "max": 6.5, "mean": 5.98, "q25": 5.73, "q75": 6.22},
        "rainfall": {"min": 90.11, "max": 119.38, "mean": 104.63, "q25": 97.48, "q75": 111.36},
    },
    "blackgram": {
        "N": {"min": 20.0, "max": 60.0, "mean": 40.02, "q25": 30.0, "q75": 50.0},
        "P": {"min": 55.0, "max": 80.0, "mean": 67.47, "q25": 60.75, "q75": 74.0},
        "K": {"min": 15.0, "max": 25.0, "mean": 19.84, "q25": 17.0, "q75": 22.0},
        "temperature": {"min": 25.04, "max": 34.93, "mean": 29.96, "q25": 27.22, "q75": 32.42},
        "humidity": {"min": 60.08, "max": 74.87, "mean": 65.12, "q25": 61.08, "q75": 68.26},
        "ph": {"min": 5.51, "max": 7.95, "mean": 6.7, "q25": 6.09, "q75": 7.3},
        "rainfall": {"min": 60.58, "max": 74.39, "mean": 67.37, "q25": 63.66, "q75": 70.89},
    },
    "chickpea": {
        "N": {"min": 20.0, "max": 60.0, "mean": 40.09, "q25": 30.0, "q75": 50.0},
        "P": {"min": 55.0, "max": 80.0, "mean": 68.05, "q25": 61.75, "q75": 75.0},
        "K": {"min": 75.0, "max": 85.0, "mean": 79.92, "q25": 77.0, "q75": 83.0},
        "temperature": {"min": 15.03, "max": 21.99, "mean": 18.87, "q25": 17.08, "q75": 20.65},
        "humidity": {"min": 14.26, "max": 19.97, "mean": 16.86, "q25": 15.56, "q75": 18.2},
        "ph": {"min": 5.51, "max": 7.97, "mean": 6.81, "q25": 6.18, "q75": 7.41},
        "rainfall": {"min": 60.5, "max": 84.79, "mean": 74.37, "q25": 68.85, "q75": 80.68},
    },
    "coconut": {
        "N": {"min": 0.0, "max": 40.0, "mean": 21.98, "q25": 11.0, "q75": 33.25},
        "P": {"min": 5.0, "max": 30.0, "mean": 16.93, "q25": 10.0, "q75": 23.0},
        "K": {"min": 25.0, "max": 35.0, "mean": 30.59, "q25": 28.0, "q75": 33.0},
        "temperature": {"min": 25.08, "max": 28.0, "mean": 27.05, "q25": 26.32, "q75": 27.71},
        "humidity": {"min": 90.03, "max": 99.98, "mean": 94.84, "q25": 92.19, "q75": 97.48},
        "ph": {"min": 5.5, "max": 6.5, "mean": 5.98, "q25": 5.72, "q75": 6.26},
        "rainfall": {"min": 131.15, "max": 225.26, "mean": 175.69, "q25": 150.52, "q75": 199.52},
    },
    "coffee": {
        "N": {"min": 80.0, "max": 120.0, "mean": 101.2, "q25": 90.75, "q75": 112.0},
        "P": {"min": 15.0, "max": 28.0, "mean": 22.31, "q25": 18.0, "q75": 27.0},
        "K": {"min": 25.0, "max": 35.0, "mean": 30.03, "q25": 27.0, "q75": 33.0},
        "temperature": {"min": 23.06, "max": 27.96, "mean": 25.54, "q25": 24.24, "q75": 26.85},
        "humidity": {"min": 50.05, "max": 69.95, "mean": 58.87, "q25": 53.43, "q75": 63.72},
        "ph": {"min": 6.0, "max": 7.0, "mean": 6.59, "q25": 6.35, "q75": 6.83},
        "rainfall": {"min": 100.32, "max": 199.33, "mean": 148.96, "q25": 121.22, "q75": 175.19},
    },
    "cotton": {
        "N": {"min": 100.0, "max": 140.0, "mean": 117.19, "q25": 107.0, "q75": 127.0},
        "P": {"min": 35.0, "max": 60.0, "mean": 46.37, "q25": 40.0, "q75": 53.0},
        "K": {"min": 15.0, "max": 25.0, "mean": 19.85, "q25": 17.0, "q75": 22.0},
        "temperature": {"min": 23.04, "max": 26.96, "mean": 23.99, "q25": 23.43, "q75": 24.34},
        "humidity": {"min": 75.05, "max": 84.84, "mean": 79.88, "q25": 77.47, "q75": 82.12},
        "ph": {"min": 5.8, "max": 8.0, "mean": 6.93, "q25": 6.39, "q75": 7.49},
        "rainfall": {"min": 60.0, "max": 99.85, "mean": 80.45, "q25": 71.0, "q75": 90.97},
    },
    "grapes": {
        "N": {"min": 0.0, "max": 40.0, "mean": 23.18, "q25": 14.0, "q75": 33.0},
        "P": {"min": 120.0, "max": 145.0, "mean": 132.11, "q25": 126.0, "q75": 138.0},
        "K": {"min": 195.0, "max": 205.0, "mean": 200.11, "q25": 197.75, "q75": 203.0},
        "temperature": {"min": 8.83, "max": 41.95, "mean": 23.86, "q25": 14.98, "q75": 32.5},
        "humidity": {"min": 80.03, "max": 83.96, "mean": 81.87, "q25": 80.88, "q75": 82.86},
        "ph": {"min": 5.51, "max": 6.95, "mean": 6.02, "q25": 5.74, "q75": 6.26},
        "rainfall": {"min": 65.87, "max": 74.96, "mean": 69.81, "q25": 67.75, "q75": 72.04},
    },
    "jute": {
        "N": {"min": 60.0, "max": 100.0, "mean": 78.4, "q25": 69.0, "q75": 88.0},
        "P": {"min": 35.0, "max": 60.0, "mean": 46.86, "q25": 40.0, "q75": 53.0},
        "K": {"min": 35.0, "max": 45.0, "mean": 39.99, "q25": 37.75, "q75": 42.0},
        "temperature": {"min": 23.99, "max": 36.93, "mean": 24.96, "q25": 24.44, "q75": 25.28},
        "humidity": {"min": 70.04, "max": 94.84, "mean": 80.35, "q25": 73.46, "q75": 87.36},
        "ph": {"min": 6.2, "max": 7.97, "mean": 6.73, "q25": 6.47, "q75": 6.97},
        "rainfall": {"min": 150.19, "max": 199.06, "mean": 174.79, "q25": 162.61, "q75": 187.47},
    },
    "kidneybeans": {
        "N": {"min": 0.0, "max": 40.0, "mean": 20.75, "q25": 11.0, "q75": 31.0},
        "P": {"min": 55.0, "max": 80.0, "mean": 68.07, "q25": 62.0, "q75": 73.0},
        "K": {"min": 15.0, "max": 25.0, "mean": 20.27, "q25": 18.0, "q75": 23.0},
        "temperature": {"min": 15.0, "max": 19.93, "mean": 17.65, "q25": 16.4, "q75": 18.94},
        "humidity": {"min": 15.1, "max": 24.79, "mean": 21.6, "q25": 19.7, "q75": 23.68},
        "ph": {"min": 5.51, "max": 7.94, "mean": 5.75, "q25": 5.6, "q75": 5.85},
        "rainfall": {"min": 60.08, "max": 99.44, "mean": 80.14, "q25": 71.0, "q75": 89.24},
    },
    "lentil": {
        "N": {"min": 0.0, "max": 40.0, "mean": 18.77, "q25": 8.0, "q75": 28.0},
        "P": {"min": 55.0, "max": 80.0, "mean": 68.36, "q25": 63.0, "q75": 74.0},
        "K": {"min": 15.0, "max": 25.0, "mean": 19.41, "q25": 17.0, "q75": 22.0},
        "temperature": {"min": 18.04, "max": 29.3, "mean": 24.51, "q25": 21.73, "q75": 27.48},
        "humidity": {"min": 24.07, "max": 64.37, "mean": 46.44, "q25": 36.98, "q75": 56.52},
        "ph": {"min": 5.86, "max": 8.78, "mean": 6.93, "q25": 6.25, "q75": 7.49},
        "rainfall": {"min": 35.08, "max": 53.62, "mean": 45.68, "q25": 41.36, "q75": 49.73},
    },
    "maize": {
        "N": {"min": 60.0, "max": 100.0, "mean": 77.76, "q25": 67.0, "q75": 88.25},
        "P": {"min": 35.0, "max": 60.0, "mean": 48.01, "q25": 42.0, "q75": 55.0},
        "K": {"min": 15.0, "max": 25.0, "mean": 19.87, "q25": 17.0, "q75": 22.0},
        "temperature": {"min": 18.04, "max": 26.94, "mean": 22.39, "q25": 20.11, "q75": 24.66},
        "humidity": {"min": 55.28, "max": 74.83, "mean": 65.63, "q25": 60.65, "q75": 70.6},
        "ph": {"min": 5.51, "max": 7.78, "mean": 6.25, "q25": 5.71, "q75": 6.58},
        "rainfall": {"min": 60.65, "max": 84.51, "mean": 72.73, "q25": 67.05, "q75": 79.11},
    },
    "mango": {
        "N": {"min": 0.0, "max": 40.0, "mean": 20.07, "q25": 10.0, "q75": 30.0},
        "P": {"min": 18.0, "max": 35.0, "mean": 27.18, "q25": 22.0, "q75": 32.0},
        "K": {"min": 25.0, "max": 35.0, "mean": 30.0, "q25": 27.0, "q75": 33.0},
        "temperature": {"min": 27.01, "max": 35.99, "mean": 31.2, "q25": 28.83, "q75": 33.31},
        "humidity": {"min": 45.02, "max": 54.96, "mean": 50.16, "q25": 47.58, "q75": 52.55},
        "ph": {"min": 4.5, "max": 7.42, "mean": 5.77, "q25": 4.99, "q75": 6.4},
        "rainfall": {"min": 89.99, "max": 99.88, "mean": 94.55, "q25": 92.15, "q75": 96.89},
    },
    "mothbeans": {
        "N": {"min": 0.0, "max": 40.0, "mean": 21.44, "q25": 11.0, "q75": 30.25},
        "P": {"min": 35.0, "max": 60.0, "mean": 48.01, "q25": 42.75, "q75": 55.0},
        "K": {"min": 15.0, "max": 25.0, "mean": 20.23, "q25": 18.0, "q75": 23.0},
        "temperature": {"min": 24.02, "max": 32.0, "mean": 28.19, "q25": 26.43, "q75": 30.12},
        "humidity": {"min": 40.01, "max": 64.96, "mean": 53.16, "q25": 46.73, "q75": 59.44},
        "ph": {"min": 3.5, "max": 9.94, "mean": 6.83, "q25": 5.38, "q75": 8.37},
        "rainfall": {"min": 30.92, "max": 74.44, "mean": 51.2, "q25": 37.99, "q75": 64.19},
    },
    "mungbean": {
        "N": {"min": 0.0, "max": 40.0, "mean": 20.99, "q25": 10.0, "q75": 31.0},
        "P": {"min": 35.0, "max": 60.0, "mean": 47.28, "q25": 40.0, "q75": 54.25},
        "K": {"min": 15.0, "max": 25.0, "mean": 19.87, "q25": 17.0, "q75": 22.0},
        "temperature": {"min": 27.01, "max": 29.91, "mean": 28.53, "q25": 27.88, "q75": 29.25},
        "humidity": {"min": 80.03, "max": 90.0, "mean": 85.5, "q25": 83.43, "q75": 87.83},
        "ph": {"min": 6.22, "max": 7.2, "mean": 6.72, "q25": 6.47, "q75": 6.98},
        "rainfall": {"min": 36.12, "max": 59.87, "mean": 48.4, "q25": 43.1, "q75": 54.84},
    },
    "muskmelon": {
        "N": {"min": 80.0, "max": 120.0, "mean": 100.32, "q25": 89.0, "q75": 111.0},
        "P": {"min": 5.0, "max": 30.0, "mean": 17.72, "q25": 12.0, "q75": 25.0},
        "K": {"min": 45.0, "max": 55.0, "mean": 50.08, "q25": 47.0, "q75": 52.25},
        "temperature": {"min": 27.02, "max": 29.94, "mean": 28.66, "q25": 27.96, "q75": 29.37},
        "humidity": {"min": 90.02, "max": 94.96, "mean": 92.34, "q25": 90.96, "q75": 93.79},
        "ph": {"min": 6.0, "max": 6.78, "mean": 6.36, "q25": 6.16, "q75": 6.55},
        "rainfall": {"min": 20.21, "max": 29.87, "mean": 24.69, "q25": 22.07, "q75": 26.86},
    },
    "orange": {
        "N": {"min": 0.0, "max": 40.0, "mean": 19.58, "q25": 9.0, "q75": 31.0},
        "P": {"min": 5.0, "max": 30.0, "mean": 16.55, "q25": 9.0, "q75": 23.0},
        "K": {"min": 5.0, "max": 15.0, "mean": 10.01, "q25": 8.0, "q75": 12.25},
        "temperature": {"min": 10.01, "max": 34.91, "mean": 22.77, "q25": 17.05, "q75": 29.55},
        "humidity": {"min": 90.01, "max": 94.96, "mean": 92.17, "q25": 91.03, "q75": 93.37},
        "ph": {"min": 6.01, "max": 8.0, "mean": 7.02, "q25": 6.48, "q75": 7.52},
        "rainfall": {"min": 100.17, "max": 119.69, "mean": 110.47, "q25": 106.22, "q75": 115.57},
    },
    "papaya": {
        "N": {"min": 31.0, "max": 70.0, "mean": 49.88, "q25": 39.0, "q75": 59.25},
        "P": {"min": 46.0, "max": 70.0, "mean": 59.05, "q25": 54.0, "q75": 65.0},
        "K": {"min": 45.0, "max": 55.0, "mean": 50.04, "q25": 47.0, "q75": 52.0},
        "temperature": {"min": 23.01, "max": 43.68, "mean": 33.72, "q25": 28.55, "q75": 39.07},
        "humidity": {"min": 90.04, "max": 94.94, "mean": 92.4, "q25": 91.17, "q75": 93.51},
        "ph": {"min": 6.5, "max": 6.99, "mean": 6.74, "q25": 6.62, "q75": 6.84},
        "rainfall": {"min": 40.35, "max": 248.86, "mean": 142.63, "q25": 81.94, "q75": 202.39},
    },
    "pigeonpeas": {
        "N": {"min": 0.0, "max": 40.0, "mean": 20.73, "q25": 10.0, "q75": 30.25},
        "P": {"min": 55.0, "max": 80.0, "mean": 67.73, "q25": 61.0, "q75": 73.25},
        "K": {"min": 15.0, "max": 25.0, "mean": 20.29, "q25": 18.0, "q75": 23.0},
        "temperature": {"min": 18.32, "max": 36.98, "mean": 27.74, "q25": 22.87, "q75": 31.47},
        "humidity": {"min": 30.4, "max": 69.69, "mean": 48.06, "q25": 38.37, "q75": 57.0},
        "ph": {"min": 4.55, "max": 7.45, "mean": 5.79, "q25": 5.0, "q75": 6.42},
        "rainfall": {"min": 90.05, "max": 198.83, "mean": 149.46, "q25": 122.35, "q75": 177.91},
    },
    "pomegranate": {
        "N": {"min": 0.0, "max": 40.0, "mean": 18.87, "q25": 8.0, "q75": 29.25},
        "P": {"min": 5.0, "max": 30.0, "mean": 18.75, "q25": 13.0, "q75": 25.0},
        "K": {"min": 35.0, "max": 45.0, "mean": 40.21, "q25": 38.0, "q75": 43.0},
        "temperature": {"min": 18.07, "max": 24.96, "mean": 21.84, "q25": 19.8, "q75": 23.89},
        "humidity": {"min": 85.13, "max": 95.0, "mean": 90.13, "q25": 88.1, "q75": 92.43},
        "ph": {"min": 5.56, "max": 7.2, "mean": 6.43, "q25": 5.97, "q75": 6.89},
        "rainfall": {"min": 102.52, "max": 112.48, "mean": 107.53, "q25": 104.97, "q75": 109.96},
    },
    "rice": {
        "N": {"min": 60.0, "max": 99.0, "mean": 79.89, "q25": 69.0, "q75": 91.0},
        "P": {"min": 35.0, "max": 60.0, "mean": 47.58, "q25": 41.0, "q75": 54.25},
        "K": {"min": 35.0, "max": 45.0, "mean": 39.87, "q25": 38.0, "q75": 42.0},
        "temperature": {"min": 20.05, "max": 26.93, "mean": 23.69, "q25": 21.93, "q75": 25.51},
        "humidity": {"min": 80.12, "max": 84.97, "mean": 82.27, "q25": 80.95, "q75": 83.47},
        "ph": {"min": 5.01, "max": 7.87, "mean": 6.43, "q25": 5.87, "q75": 7.04},
        "rainfall": {"min": 182.56, "max": 298.56, "mean": 236.18, "q25": 204.33, "q75": 264.13},
    },
    "watermelon": {
        "N": {"min": 80.0, "max": 120.0, "mean": 99.42, "q25": 89.0, "q75": 110.0},
        "P": {"min": 5.0, "max": 30.0, "mean": 17.0, "q25": 10.0, "q75": 23.0},
        "K": {"min": 45.0, "max": 55.0, "mean": 50.22, "q25": 47.0, "q75": 53.0},
        "temperature": {"min": 24.04, "max": 26.99, "mean": 25.59, "q25": 24.9, "q75": 26.26},
        "humidity": {"min": 80.03, "max": 89.98, "mean": 85.16, "q25": 83.01, "q75": 87.64},
        "ph": {"min": 6.0, "max": 6.96, "mean": 6.5, "q25": 6.26, "q75": 6.76},
        "rainfall": {"min": 40.13, "max": 59.76, "mean": 50.79, "q25": 46.06, "q75": 56.41},
    },
}

# Feature display names and units for readable reports
FEATURE_DISPLAY: dict[str, dict] = {
    "N":           {"name": "Nitrogen (N)", "unit": "kg/ha"},
    "P":           {"name": "Phosphorus (P)", "unit": "kg/ha"},
    "K":           {"name": "Potassium (K)", "unit": "kg/ha"},
    "temperature": {"name": "Temperature", "unit": "°C"},
    "humidity":    {"name": "Humidity", "unit": "%"},
    "ph":          {"name": "Soil pH", "unit": ""},
    "rainfall":    {"name": "Rainfall", "unit": "mm"},
}


# ═══════════════════════════════════════════════════════
# Soil Analyzer
# ═══════════════════════════════════════════════════════

class SoilAnalyzer:
    """
    Analyzes soil and climate inputs to produce:
    - Moisture level estimation
    - Mineral profile assessment (NPK)
    - Overall fertility score
    - Crop-mineral suitability matrix
    - Planting advisory with reasoning
    - Soil improvement tips
    """

    # ── Moisture level from humidity + rainfall ─────────
    @staticmethod
    def assess_moisture(humidity: float, rainfall: float) -> dict:
        """
        Estimate soil moisture from humidity and rainfall.
        Returns level (Low/Moderate/High/Very High), percentage, and description.
        """
        # Weighted composite: rainfall has more impact on soil moisture
        moisture_score = (humidity * 0.35) + (rainfall / 300 * 100 * 0.65)
        moisture_score = min(100, max(0, moisture_score))

        if moisture_score < 25:
            level = "Low"
            desc = "Soil moisture is low. Suitable for drought-tolerant crops like mothbeans, lentil, and chickpea."
        elif moisture_score < 50:
            level = "Moderate"
            desc = "Moderate soil moisture. Supports a wide variety of crops including maize, cotton, and coffee."
        elif moisture_score < 75:
            level = "High"
            desc = "High soil moisture. Ideal for crops like rice, banana, and jute that need ample water."
        else:
            level = "Very High"
            desc = "Very high soil moisture. Best for water-loving crops like rice and jute. Avoid drought-tolerant crops."

        return {
            "level": level,
            "percentage": round(moisture_score, 1),
            "description": desc,
        }

    # ── Mineral assessment ──────────────────────────────
    @staticmethod
    def assess_mineral(value: float, feature: str) -> dict:
        """
        Rate a single mineral/feature as Deficient/Low/Optimal/High/Excess.
        Uses global distribution from the dataset.
        """
        # Global ranges for NPK and other features
        ranges = {
            "N":           {"deficient": 10, "low": 30, "optimal_min": 30, "optimal_max": 90, "high": 120, "excess": 140},
            "P":           {"deficient": 15, "low": 30, "optimal_min": 30, "optimal_max": 80, "high": 120, "excess": 145},
            "K":           {"deficient": 10, "low": 20, "optimal_min": 20, "optimal_max": 80, "high": 150, "excess": 205},
            "temperature": {"deficient": 10, "low": 15, "optimal_min": 18, "optimal_max": 35, "high": 40, "excess": 44},
            "humidity":    {"deficient": 20, "low": 40, "optimal_min": 50, "optimal_max": 90, "high": 95, "excess": 100},
            "ph":          {"deficient": 4.0, "low": 5.0, "optimal_min": 5.5, "optimal_max": 7.5, "high": 8.5, "excess": 10.0},
            "rainfall":    {"deficient": 30, "low": 50, "optimal_min": 60, "optimal_max": 200, "high": 250, "excess": 300},
        }

        r = ranges.get(feature, ranges["N"])

        if value <= r["deficient"]:
            status = "Deficient"
            color = "red"
        elif value <= r["low"]:
            status = "Low"
            color = "orange"
        elif value <= r["optimal_max"]:
            status = "Optimal"
            color = "green"
        elif value <= r["high"]:
            status = "High"
            color = "amber"
        else:
            status = "Excess"
            color = "red"

        return {
            "value": round(value, 2),
            "status": status,
            "color": color,
            "unit": FEATURE_DISPLAY.get(feature, {}).get("unit", ""),
            "name": FEATURE_DISPLAY.get(feature, {}).get("name", feature),
        }

    # ── Fertility score ─────────────────────────────────
    @staticmethod
    def compute_fertility_score(
        n: float, p: float, k: float,
        ph: float, humidity: float, rainfall: float,
    ) -> dict:
        """
        Composite fertility score (0–100) based on NPK balance, pH, and moisture.
        """
        # NPK balance — score higher when all three are present
        npk_total = n + p + k
        npk_score = min(100, (npk_total / 200) * 100) if npk_total > 0 else 0

        # Balance bonus — reward when NPK are balanced
        if npk_total > 0:
            proportions = [n / npk_total, p / npk_total, k / npk_total]
            entropy = -sum(p * np.log2(p + 1e-10) for p in proportions)
            max_entropy = np.log2(3)  # max when perfectly balanced
            balance_score = (entropy / max_entropy) * 100
        else:
            balance_score = 0

        # pH score — peaks near 6.5 (ideal for most crops)
        ph_distance = abs(ph - 6.5)
        ph_score = max(0, 100 - (ph_distance / 3.5) * 100)

        # Moisture score
        moisture_composite = (humidity * 0.35) + (rainfall / 300 * 100 * 0.65)
        moisture_score = min(100, max(0, moisture_composite))

        # Weighted composite
        fertility = (
            npk_score * 0.35 +
            balance_score * 0.15 +
            ph_score * 0.25 +
            moisture_score * 0.25
        )
        fertility = round(min(100, max(0, fertility)), 1)

        if fertility >= 80:
            grade = "Excellent"
            desc = "Your soil has excellent fertility with well-balanced nutrients and favorable conditions."
        elif fertility >= 60:
            grade = "Good"
            desc = "Soil fertility is good. Minor adjustments to nutrients may improve yields."
        elif fertility >= 40:
            grade = "Moderate"
            desc = "Moderate fertility. Consider soil amendments to improve nutrient levels."
        elif fertility >= 20:
            grade = "Low"
            desc = "Low soil fertility. Significant amendments needed for most crops."
        else:
            grade = "Very Low"
            desc = "Very low soil fertility. Major soil improvement required before planting."

        return {
            "score": fertility,
            "grade": grade,
            "description": desc,
            "breakdown": {
                "npk_score": round(npk_score, 1),
                "balance_score": round(balance_score, 1),
                "ph_score": round(ph_score, 1),
                "moisture_score": round(moisture_score, 1),
            },
        }

    # ── Crop suitability matching ───────────────────────
    @staticmethod
    def compute_crop_suitability(
        crop: str,
        n: float, p: float, k: float,
        temperature: float, humidity: float,
        ph: float, rainfall: float,
    ) -> dict:
        """
        Compare user's soil values against the crop's ideal ranges.
        Returns per-feature match info + overall suitability score.
        """
        ideal = CROP_IDEAL_RANGES.get(crop)
        if not ideal:
            return {
                "crop": crop,
                "overall_match": 0,
                "verdict": "unknown",
                "features": [],
                "tips": [],
            }

        user_values = {
            "N": n, "P": p, "K": k,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph, "rainfall": rainfall,
        }

        feature_matches = []
        tips = []
        match_scores = []

        for feat, val in user_values.items():
            if feat not in ideal:
                continue

            crop_range = ideal[feat]
            q25 = crop_range["q25"]
            q75 = crop_range["q75"]
            fmin = crop_range["min"]
            fmax = crop_range["max"]
            mean = crop_range["mean"]
            display = FEATURE_DISPLAY.get(feat, {"name": feat, "unit": ""})

            # Calculate match score (0-100)
            if q25 <= val <= q75:
                # In ideal range
                score = 100
                status = "Ideal"
                color = "green"
            elif fmin <= val <= fmax:
                # In tolerable range
                if val < q25:
                    score = max(40, 100 - ((q25 - val) / (q25 - fmin + 0.01)) * 60)
                else:
                    score = max(40, 100 - ((val - q75) / (fmax - q75 + 0.01)) * 60)
                status = "Acceptable"
                color = "amber"
            else:
                # Outside tolerable range
                if val < fmin:
                    overshoot = (fmin - val) / (fmin + 0.01)
                else:
                    overshoot = (val - fmax) / (fmax + 0.01)
                score = max(0, 40 - overshoot * 80)
                status = "Poor"
                color = "red"

            match_scores.append(score)

            feature_matches.append({
                "feature": feat,
                "name": display["name"],
                "unit": display["unit"],
                "your_value": round(val, 2),
                "ideal_min": q25,
                "ideal_max": q75,
                "tolerable_min": fmin,
                "tolerable_max": fmax,
                "score": round(score, 1),
                "status": status,
                "color": color,
            })

            # Generate tips for poor or acceptable matches
            if status == "Poor":
                if val < fmin:
                    diff = round(q25 - val, 1)
                    tips.append(
                        f"⚠️ {display['name']} is too low ({val} {display['unit']}). "
                        f"Increase by ~{diff} {display['unit']} to reach the ideal range "
                        f"({q25}–{q75} {display['unit']}) for {crop}."
                    )
                else:
                    diff = round(val - q75, 1)
                    tips.append(
                        f"⚠️ {display['name']} is too high ({val} {display['unit']}). "
                        f"Reduce by ~{diff} {display['unit']} to reach the ideal range "
                        f"({q25}–{q75} {display['unit']}) for {crop}."
                    )
            elif status == "Acceptable" and score < 70:
                if val < q25:
                    diff = round(q25 - val, 1)
                    tips.append(
                        f"💡 Consider increasing {display['name']} by ~{diff} {display['unit']} "
                        f"for optimal {crop} growth (ideal: {q25}–{q75} {display['unit']})."
                    )
                else:
                    diff = round(val - q75, 1)
                    tips.append(
                        f"💡 Consider reducing {display['name']} by ~{diff} {display['unit']} "
                        f"for optimal {crop} growth (ideal: {q25}–{q75} {display['unit']})."
                    )

        # Overall match = weighted average (NPK features weighted higher)
        weights = {"N": 1.2, "P": 1.2, "K": 1.2, "temperature": 1.0,
                   "humidity": 1.0, "ph": 1.1, "rainfall": 1.0}
        weighted_sum = sum(
            fm["score"] * weights.get(fm["feature"], 1.0)
            for fm in feature_matches
        )
        total_weight = sum(
            weights.get(fm["feature"], 1.0) for fm in feature_matches
        )
        overall = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0

        # Verdict
        if overall >= 80:
            verdict = "excellent"
            verdict_text = f"✅ Excellent match! Your soil is highly suitable for {crop}."
            should_plant = True
        elif overall >= 60:
            verdict = "good"
            verdict_text = f"✅ Good match. Your soil conditions are favorable for {crop} with minor adjustments."
            should_plant = True
        elif overall >= 40:
            verdict = "moderate"
            verdict_text = f"⚠️ Moderate match. {crop.capitalize()} can grow but may need soil amendments."
            should_plant = False
        else:
            verdict = "poor"
            verdict_text = f"❌ Poor match. Your soil is not well-suited for {crop}. Consider alternative crops."
            should_plant = False

        return {
            "crop": crop,
            "emoji": CROP_EMOJIS.get(crop, "🌱"),
            "overall_match": overall,
            "verdict": verdict,
            "verdict_text": verdict_text,
            "should_plant": should_plant,
            "features": feature_matches,
            "tips": tips,
        }

    # ── Full analysis ───────────────────────────────────
    @classmethod
    def analyze(
        cls,
        n: float, p: float, k: float,
        temperature: float, humidity: float,
        ph: float, rainfall: float,
        top_crops: list[str],
    ) -> dict:
        """
        Run full soil analysis and return structured result.
        """
        # Moisture
        moisture = cls.assess_moisture(humidity, rainfall)

        # Mineral profile for NPK + pH
        minerals = {
            "N": cls.assess_mineral(n, "N"),
            "P": cls.assess_mineral(p, "P"),
            "K": cls.assess_mineral(k, "K"),
            "ph": cls.assess_mineral(ph, "ph"),
            "temperature": cls.assess_mineral(temperature, "temperature"),
            "humidity": cls.assess_mineral(humidity, "humidity"),
            "rainfall": cls.assess_mineral(rainfall, "rainfall"),
        }

        # Fertility score
        fertility = cls.compute_fertility_score(n, p, k, ph, humidity, rainfall)

        # Crop suitability for top crops
        crop_suitability = []
        for crop in top_crops:
            suitability = cls.compute_crop_suitability(
                crop, n, p, k, temperature, humidity, ph, rainfall,
            )
            crop_suitability.append(suitability)

        return {
            "moisture": moisture,
            "minerals": minerals,
            "fertility": fertility,
            "crop_suitability": crop_suitability,
        }


# ═══════════════════════════════════════════════════════
# Soil analyzer singleton
# ═══════════════════════════════════════════════════════
soil_analyzer = SoilAnalyzer()


def _make_cache_key(n, p, k, temperature, humidity, ph, rainfall) -> tuple:
    """Round inputs to create a hashable cache key."""
    return (
        round(n, 2), round(p, 2), round(k, 2),
        round(temperature, 2), round(humidity, 2),
        round(ph, 2), round(rainfall, 2),
    )


class CropModel:
    """Singleton wrapper around the trained model + preprocessing pipeline."""

    def __init__(self) -> None:
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._feature_cols: list[str] = []
        self._loaded = False
        self._load_time_ms: float = 0
        self._prediction_cache: dict = {}
        self._cache_max_size = 500

    # ── Loading ─────────────────────────────────────────
    def load(self) -> None:
        """Load model and pipeline from disk."""
        if self._loaded:
            return

        model_path = Path(MODEL_PATH)
        pipe_path = Path(PIPELINE_PATH)

        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not pipe_path.exists():
            logger.error("Pipeline file not found: %s", pipe_path)
            raise FileNotFoundError(f"Pipeline file not found: {pipe_path}")

        t0 = time.perf_counter()

        self._model = joblib.load(model_path)
        pipeline = joblib.load(pipe_path)

        self._scaler = pipeline["scaler"]
        self._label_encoder = pipeline["label_encoder"]
        self._feature_cols = pipeline["feature_cols"]
        self._loaded = True
        self._load_time_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "✅ Model loaded in %.1fms — %s (n_classes=%d, features=%s)",
            self._load_time_ms,
            type(self._model).__name__,
            len(self._label_encoder.classes_),
            self._feature_cols,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def classes(self) -> list[str]:
        if not self._loaded:
            return []
        return list(self._label_encoder.classes_)

    # ── Prediction ──────────────────────────────────────
    def predict(
        self,
        n: float, p: float, k: float,
        temperature: float, humidity: float,
        ph: float, rainfall: float,
    ) -> dict:
        """
        Run inference and return structured result with soil analysis.

        Returns dict with keys:
            recommended_crop, confidence, emoji, info,
            top3, all_probabilities, input_summary,
            soil_analysis (NEW)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Check cache
        cache_key = _make_cache_key(n, p, k, temperature, humidity, ph, rainfall)
        if cache_key in self._prediction_cache:
            logger.debug("Cache hit for input: %s", cache_key)
            return self._prediction_cache[cache_key]

        t0 = time.perf_counter()

        # Build input DataFrame in the same column order as training
        raw = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        df = pd.DataFrame(raw, columns=self._feature_cols)

        # Scale
        scaled = self._scaler.transform(df)

        # Predict probabilities
        proba = self._model.predict_proba(scaled)[0]

        # Top-3
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_crops = self._label_encoder.inverse_transform(top3_idx)
        top3_probs = proba[top3_idx]

        best = top3_crops[0]

        # All probabilities
        all_crops = self._label_encoder.inverse_transform(np.arange(len(proba)))
        all_probs = {
            crop: round(float(prob) * 100, 2)
            for crop, prob in zip(all_crops, proba)
        }

        # ── Soil Analysis (NEW) ─────────────────────────
        soil_analysis = soil_analyzer.analyze(
            n=n, p=p, k=k,
            temperature=temperature,
            humidity=humidity,
            ph=ph, rainfall=rainfall,
            top_crops=list(top3_crops),
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "🌾 Prediction: %s (%.1f%%) in %.1fms | Fertility: %s (%.0f)",
            best, top3_probs[0] * 100, elapsed_ms,
            soil_analysis["fertility"]["grade"],
            soil_analysis["fertility"]["score"],
        )

        result = {
            "recommended_crop": best,
            "confidence": round(float(top3_probs[0]) * 100, 2),
            "emoji": CROP_EMOJIS.get(best, "🌱"),
            "info": CROP_INFO.get(best, ""),
            "top3": [
                {
                    "crop": crop,
                    "probability": round(float(prob) * 100, 2),
                    "emoji": CROP_EMOJIS.get(crop, "🌱"),
                    "info": CROP_INFO.get(crop, ""),
                }
                for crop, prob in zip(top3_crops, top3_probs)
            ],
            "all_probabilities": all_probs,
            "input_summary": {
                "N": n, "P": p, "K": k,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            },
            "soil_analysis": soil_analysis,
        }

        # Cache result (evict oldest if full)
        if len(self._prediction_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._prediction_cache))
            del self._prediction_cache[oldest_key]
        self._prediction_cache[cache_key] = result

        return result

    # ── Model info ──────────────────────────────────────
    def get_info(self) -> dict:
        """Return model metadata for the About / API info endpoint."""
        return {
            "model_name": "Random Forest",
            "n_estimators": getattr(self._model, "n_estimators", None),
            "test_accuracy": 99.55,
            "train_accuracy": 100.0,
            "f1_macro": 99.55,
            "cv_accuracy": 99.45,
            "cv_std": 0.27,
            "n_classes": len(self.classes),
            "classes": self.classes,
            "features": self._feature_cols,
            "feature_importance": [
                {"rank": 1, "feature": "humidity",    "avg_rank": 1.33},
                {"rank": 2, "feature": "rainfall",    "avg_rank": 2.33},
                {"rank": 3, "feature": "K",           "avg_rank": 3.00},
                {"rank": 4, "feature": "N",           "avg_rank": 3.67},
                {"rank": 5, "feature": "P",           "avg_rank": 4.67},
                {"rank": 6, "feature": "temperature", "avg_rank": 6.33},
                {"rank": 7, "feature": "ph",          "avg_rank": 6.67},
            ],
        }


# Module-level singleton
crop_model = CropModel()
