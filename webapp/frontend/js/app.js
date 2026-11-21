/**
 * 🌾 Crop Recommendation System — Frontend Logic
 * Phase 7: Premium UI with canvas charts, animations, confetti
 *
 * Handles: form validation, API calls, result rendering,
 * donut chart, probability chart, range indicators,
 * animated counters, confetti, session history, and UI state.
 */

// ═══════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════

const API_BASE = '';  // Same origin — no prefix needed.

const SCENARIOS = {
    rice:       { N: 80,  P: 48,  K: 40,  temperature: 23.5, humidity: 82.0, ph: 6.5,  rainfall: 236.0 },
    apple:      { N: 20,  P: 130, K: 200, temperature: 22.6, humidity: 92.0, ph: 5.9,  rainfall: 113.0 },
    maize:      { N: 80,  P: 48,  K: 20,  temperature: 22.4, humidity: 65.0, ph: 6.2,  rainfall: 68.0  },
    coffee:     { N: 100, P: 18,  K: 30,  temperature: 25.5, humidity: 58.0, ph: 6.8,  rainfall: 158.0 },
    banana:     { N: 100, P: 82,  K: 50,  temperature: 27.0, humidity: 80.0, ph: 6.0,  rainfall: 105.0 },
    chickpea:   { N: 40,  P: 68,  K: 80,  temperature: 18.0, humidity: 17.0, ph: 7.0,  rainfall: 75.0  },
    coconut:    { N: 22,  P: 17,  K: 31,  temperature: 27.4, humidity: 94.8, ph: 6.0,  rainfall: 175.7 },
    grapes:     { N: 23,  P: 132, K: 200, temperature: 23.8, humidity: 81.9, ph: 6.0,  rainfall: 69.6  },
    cotton:     { N: 118, P: 46,  K: 20,  temperature: 24.0, humidity: 79.8, ph: 6.9,  rainfall: 80.4  },
    watermelon: { N: 99,  P: 17,  K: 50,  temperature: 25.6, humidity: 85.2, ph: 6.5,  rainfall: 50.8  },
    mango:      { N: 20,  P: 27,  K: 30,  temperature: 31.2, humidity: 50.2, ph: 5.8,  rainfall: 94.7  },
    lentil:     { N: 19,  P: 68,  K: 19,  temperature: 24.5, humidity: 64.8, ph: 6.9,  rainfall: 45.6  },
};

const FIELD_MAP = [
    { id: 'input-n',    key: 'N',           label: 'Nitrogen (N)',   min: 0,   max: 140, rangeId: 'range-n'    },
    { id: 'input-p',    key: 'P',           label: 'Phosphorus (P)', min: 5,   max: 145, rangeId: 'range-p'   },
    { id: 'input-k',    key: 'K',           label: 'Potassium (K)', min: 5,   max: 205, rangeId: 'range-k'    },
    { id: 'input-temp', key: 'temperature', label: 'Temperature',   min: 8,   max: 44,  rangeId: 'range-temp' },
    { id: 'input-hum',  key: 'humidity',    label: 'Humidity',      min: 14,  max: 100, rangeId: 'range-hum'  },
    { id: 'input-ph',   key: 'ph',          label: 'Soil pH',      min: 3.5, max: 10,  rangeId: 'range-ph'   },
    { id: 'input-rain', key: 'rainfall',    label: 'Rainfall',     min: 20,  max: 300, rangeId: 'range-rain'  },
];

const HISTORY_KEY = 'crop_recommendation_history';

const CHART_COLORS = [
    '#43cea2', '#4facfe', '#00f2fe', '#fbbf24',
    '#a78bfa', '#f472b6', '#fb923c', '#34d399',
];


// ═══════════════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════════════

const form            = document.getElementById('predict-form');
const submitBtn       = document.getElementById('submit-btn');
const errorBanner     = document.getElementById('error-banner');
const resultsSection  = document.getElementById('results-section');
const scenarioSelect  = document.getElementById('scenario-select');
const hamburger       = document.getElementById('nav-hamburger');
const navLinks        = document.getElementById('nav-links');
const historyToggle   = document.getElementById('history-toggle');
const historyContent  = document.getElementById('history-content');
const historyList     = document.getElementById('history-list');
const historyCount    = document.getElementById('history-count');
const historyClear    = document.getElementById('history-clear');
const newPredBtn      = document.getElementById('new-prediction-btn');
const celebrationEl   = document.getElementById('celebration-overlay');


// ═══════════════════════════════════════════════════════
// Range Indicators — show input position visually
// ═══════════════════════════════════════════════════════

function updateRangeIndicator(field) {
    const el = document.getElementById(field.id);
    const fill = document.getElementById(field.rangeId);
    if (!el || !fill) return;

    const val = parseFloat(el.value);
    if (isNaN(val)) {
        fill.style.width = '0%';
        return;
    }
    const pct = Math.max(0, Math.min(100, ((val - field.min) / (field.max - field.min)) * 100));
    fill.style.width = `${pct}%`;
}

// Attach input listeners for range indicators
FIELD_MAP.forEach(f => {
    const el = document.getElementById(f.id);
    if (el) {
        el.addEventListener('input', () => updateRangeIndicator(f));
    }
});


// ═══════════════════════════════════════════════════════
// Scenario Quick-Fill
// ═══════════════════════════════════════════════════════

scenarioSelect.addEventListener('change', () => {
    const key = scenarioSelect.value;
    if (!key || !SCENARIOS[key]) return;

    const data = SCENARIOS[key];
    FIELD_MAP.forEach(f => {
        const el = document.getElementById(f.id);
        if (el && data[f.key] !== undefined) {
            el.value = data[f.key];
            el.classList.remove('error');
            const errEl = document.getElementById(`error-${f.id.replace('input-', '')}`);
            if (errEl) errEl.textContent = '';
            updateRangeIndicator(f);
        }
    });
});

// ═══════════════════════════════════════════════════════
// Field Variance Perturbation (±10%)
// ═══════════════════════════════════════════════════════

const jitterBtn = document.getElementById('jitter-btn');
if (jitterBtn) {
    jitterBtn.addEventListener('click', () => {
        FIELD_MAP.forEach(f => {
            const el = document.getElementById(f.id);
            if (!el) return;
            let val = parseFloat(el.value);
            if (isNaN(val)) {
                // If empty, fill with middle value
                val = (f.min + f.max) / 2;
            }
            // Add ±10% random Gaussian jitter
            const span = (f.max - f.min) * 0.08;
            const delta = (Math.random() - 0.5) * 2 * span;
            let newVal = val + delta;
            newVal = Math.max(f.min, Math.min(f.max, newVal));
            // Step precision
            el.value = f.id === 'input-ph' ? newVal.toFixed(2) : f.id === 'input-temp' || f.id === 'input-hum' ? newVal.toFixed(1) : Math.round(newVal);
            el.classList.remove('error');
            const errEl = document.getElementById(`error-${f.id.replace('input-', '')}`);
            if (errEl) errEl.textContent = '';
            updateRangeIndicator(f);
        });

        // Flash button feedback
        jitterBtn.style.transform = 'scale(0.96)';
        setTimeout(() => { jitterBtn.style.transform = ''; }, 150);
    });
}


// ═══════════════════════════════════════════════════════
// Form Validation
// ═══════════════════════════════════════════════════════

function validateForm() {
    let isValid = true;

    FIELD_MAP.forEach(f => {
        const el = document.getElementById(f.id);
        const errorId = `error-${f.id.replace('input-', '')}`;
        const errEl = document.getElementById(errorId);
        const val = el.value.trim();

        // Clear previous
        el.classList.remove('error');
        if (errEl) errEl.textContent = '';

        if (val === '') {
            el.classList.add('error');
            if (errEl) errEl.textContent = `${f.label} is required.`;
            isValid = false;
        } else {
            const num = parseFloat(val);
            if (isNaN(num)) {
                el.classList.add('error');
                if (errEl) errEl.textContent = `Enter a valid number.`;
                isValid = false;
            } else if (num < f.min || num > f.max) {
                el.classList.add('error');
                if (errEl) errEl.textContent = `Must be between ${f.min} and ${f.max}.`;
                isValid = false;
            }
        }
    });

    return isValid;
}


// ═══════════════════════════════════════════════════════
// API Call
// ═══════════════════════════════════════════════════════

async function fetchPrediction(payload) {
    const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => null);
        if (res.status === 422 && err?.detail) {
            const messages = err.detail.map(d => {
                const field = d.loc?.[d.loc.length - 1] || 'Unknown field';
                return `${field}: ${d.msg}`;
            });
            throw new Error(messages.join('\n'));
        }
        throw new Error(err?.detail || `Server error (${res.status})`);
    }

    return await res.json();
}


// ═══════════════════════════════════════════════════════
// Animated Counter
// ═══════════════════════════════════════════════════════

function animateCounter(element, target, suffix = '%', duration = 800) {
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (target - start) * eased;

        element.textContent = `Confidence: ${current.toFixed(1)}${suffix}`;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}


// ═══════════════════════════════════════════════════════
// Canvas Donut Chart
// ═══════════════════════════════════════════════════════

function drawDonutChart(top3Data) {
    const canvas = document.getElementById('donut-chart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const size = 220;

    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2;
    const outerR = 90;
    const innerR = 55;

    // Clear
    ctx.clearRect(0, 0, size, size);

    // Calculate total and angles
    const total = top3Data.reduce((s, d) => s + d.probability, 0);
    if (total <= 0) return;

    let startAngle = -Math.PI / 2;
    const colors = ['#43cea2', '#4facfe', '#64748b'];

    // Animation
    let animProgress = 0;
    const animDuration = 1000;
    const animStart = performance.now();

    function animate(now) {
        const elapsed = now - animStart;
        animProgress = Math.min(elapsed / animDuration, 1);
        const eased = 1 - Math.pow(1 - animProgress, 3);

        ctx.clearRect(0, 0, size, size);

        // Background ring
        ctx.beginPath();
        ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
        ctx.arc(cx, cy, innerR, Math.PI * 2, 0, true);
        ctx.fillStyle = 'rgba(255,255,255,0.03)';
        ctx.fill();

        // Data arcs
        let currentAngle = -Math.PI / 2;
        top3Data.forEach((d, i) => {
            const sweep = (d.probability / total) * Math.PI * 2 * eased;
            const gap = 0.03;

            ctx.beginPath();
            ctx.arc(cx, cy, outerR, currentAngle + gap, currentAngle + sweep - gap);
            ctx.arc(cx, cy, innerR, currentAngle + sweep - gap, currentAngle + gap, true);
            ctx.closePath();
            ctx.fillStyle = colors[i] || colors[2];
            ctx.fill();

            currentAngle += sweep;
        });

        // Center text
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#f1f5f9';
        ctx.font = '700 22px Inter, sans-serif';
        ctx.fillText(`${(top3Data[0].probability * eased).toFixed(1)}%`, cx, cy - 6);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '500 10px Inter, sans-serif';
        ctx.fillText('TOP PICK', cx, cy + 14);

        if (animProgress < 1) {
            requestAnimationFrame(animate);
        } else {
            // Draw legend
            let ly = size - 8;
            const legendX = cx;
            ctx.textAlign = 'center';
            top3Data.forEach((d, i) => {
                // Skip — legend is rendered in HTML for accessibility
            });
        }
    }

    requestAnimationFrame(animate);
}


// ═══════════════════════════════════════════════════════
// Confetti Celebration
// ═══════════════════════════════════════════════════════

function launchConfetti() {
    if (!celebrationEl) return;

    const colors = ['#43cea2', '#4facfe', '#00f2fe', '#fbbf24', '#a78bfa', '#f472b6', '#34d399'];
    const count = 50;

    for (let i = 0; i < count; i++) {
        const piece = document.createElement('div');
        piece.className = 'confetti-piece';
        piece.style.left = `${Math.random() * 100}%`;
        piece.style.top = `-${Math.random() * 20}px`;
        piece.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        piece.style.width = `${6 + Math.random() * 6}px`;
        piece.style.height = `${6 + Math.random() * 6}px`;
        piece.style.borderRadius = Math.random() > 0.5 ? '50%' : '2px';
        piece.style.animationDuration = `${1.5 + Math.random() * 2}s`;
        piece.style.animationDelay = `${Math.random() * 0.5}s`;
        celebrationEl.appendChild(piece);
    }

    // Cleanup after animation
    setTimeout(() => {
        celebrationEl.innerHTML = '';
    }, 4000);
}


// ═══════════════════════════════════════════════════════
// Result Rendering
// ═══════════════════════════════════════════════════════

function renderResults(data, inputValues) {
    // Primary result
    document.getElementById('result-emoji').textContent = data.emoji;
    document.getElementById('result-crop').textContent = data.recommended_crop;
    document.getElementById('result-info').textContent = data.info ? `💡 ${data.info}` : '';

    // Animated confidence counter
    const confEl = document.getElementById('result-confidence');
    animateCounter(confEl, data.confidence, '%', 900);

    // Confidence bar (animate after a tick)
    const bar = document.getElementById('result-bar');
    bar.style.width = '0%';
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            bar.style.width = `${data.confidence}%`;
        });
    });

    // Donut chart
    drawDonutChart(data.top3);

    // Runner-ups / Alternative Viable Crops
    const runnerContainer = document.getElementById('runner-ups-container');
    const runnerEl = document.getElementById('runner-ups');
    if (runnerContainer && runnerEl) {
        const runners = (data.top3 || []).slice(1);
        if (runners.length > 0) {
            runnerEl.innerHTML = runners.map((r, i) => `
                <div class="runner-card">
                    <span style="font-size: 0.72rem; color: var(--c-accent); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">#${i + 2} Alternative Option</span>
                    <div class="runner-emoji">${r.emoji || '🌱'}</div>
                    <div class="runner-name">${r.crop}</div>
                    <div class="runner-prob">${r.probability}% Match Confidence</div>
                    <div style="font-size: 0.78rem; color: var(--c-text-muted); margin-top: 6px; line-height: 1.35;">
                        ${r.info || 'Viable alternative candidate for this soil and climate profile.'}
                    </div>
                </div>
            `).join('');
            runnerContainer.style.display = 'block';
        } else {
            runnerContainer.style.display = 'none';
        }
    }

    // Probability chart
    renderProbChart(data.all_probabilities, data.recommended_crop);

    // Input summary
    document.getElementById('input-summary').innerHTML = `
        <strong>📋 Your Input Summary</strong><br>
        N = ${inputValues.N} kg/ha · P = ${inputValues.P} kg/ha · K = ${inputValues.K} kg/ha ·
        Temperature = ${inputValues.temperature}°C · Humidity = ${inputValues.humidity}% ·
        pH = ${inputValues.ph} · Rainfall = ${inputValues.rainfall} mm`;

    // Show results section
    resultsSection.style.display = '';

    // ── Soil Analysis Dashboard ──────────────────
    if (data.soil_analysis) {
        renderSoilAnalysis(data.soil_analysis);
    }

    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Celebrate high confidence
    if (data.confidence >= 95) {
        setTimeout(launchConfetti, 400);
    }
}


// ═══════════════════════════════════════════════════════
// Soil Analysis Dashboard Rendering
// ═══════════════════════════════════════════════════════

function renderSoilAnalysis(analysis) {
    renderMoisture(analysis.moisture);
    renderFertility(analysis.fertility);
    renderMinerals(analysis.minerals);
    renderCropSuitability(analysis.crop_suitability);
}


// ── Moisture Gauge ─────────────────────────────────────

function renderMoisture(moisture) {
    const pct = moisture.percentage;
    const circumference = 314.16; // 2 * PI * 50
    const offset = circumference - (pct / 100) * circumference;

    // Ring fill
    const ring = document.getElementById('moisture-ring-fill');
    if (ring) {
        ring.style.strokeDashoffset = circumference; // reset
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                ring.style.strokeDashoffset = offset;
            });
        });
    }

    // Percentage text
    const pctEl = document.getElementById('moisture-percentage');
    if (pctEl) pctEl.textContent = `${pct}%`;

    // Level badge
    const levelEl = document.getElementById('moisture-level');
    if (levelEl) {
        levelEl.textContent = moisture.level;
        levelEl.className = 'soil-moisture-level';
        const levelKey = moisture.level.toLowerCase().replace(/\s/g, '');
        levelEl.classList.add(`moisture-level-${levelKey}`);
    }

    // Description
    const descEl = document.getElementById('moisture-desc');
    if (descEl) descEl.textContent = moisture.description;
}


// ── Fertility Score ────────────────────────────────────

function renderFertility(fertility) {
    const score = fertility.score;
    const circumference = 314.16;
    const offset = circumference - (score / 100) * circumference;

    // Ring fill
    const ring = document.getElementById('fertility-ring-fill');
    if (ring) {
        ring.style.strokeDashoffset = circumference;
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                ring.style.strokeDashoffset = offset;
            });
        });
    }

    // Score text
    const scoreEl = document.getElementById('fertility-score');
    if (scoreEl) scoreEl.textContent = Math.round(score);

    // Grade badge
    const gradeEl = document.getElementById('fertility-grade');
    if (gradeEl) {
        gradeEl.textContent = fertility.grade;
        gradeEl.className = 'fertility-grade';
        const gradeKey = fertility.grade.toLowerCase().replace(/\s/g, '');
        gradeEl.classList.add(`fertility-${gradeKey}`);
    }

    // Description
    const descEl = document.getElementById('fertility-desc');
    if (descEl) descEl.textContent = fertility.description;
}


// ── Mineral Profile Cards ──────────────────────────────

function renderMinerals(minerals) {
    const grid = document.getElementById('mineral-grid');
    if (!grid) return;
    grid.innerHTML = '';

    // Order: N, P, K, pH, Temperature, Humidity, Rainfall
    const order = ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall'];

    order.forEach(key => {
        const m = minerals[key];
        if (!m) return;

        const badgeClass = {
            'green': 'mineral-badge-green',
            'amber': 'mineral-badge-amber',
            'red':   'mineral-badge-red',
            'orange':'mineral-badge-orange',
        }[m.color] || 'mineral-badge-amber';

        const card = document.createElement('div');
        card.className = 'mineral-card';
        card.setAttribute('data-status', m.status);
        card.innerHTML = `
            <div class="mineral-card-name">${m.name}</div>
            <div class="mineral-card-value">
                ${m.value} <span class="mineral-card-unit">${m.unit}</span>
            </div>
            <span class="mineral-card-badge ${badgeClass}">${m.status}</span>
        `;
        grid.appendChild(card);
    });
}


// ── Crop Suitability & Planting Advisory ───────────────

function renderCropSuitability(suitabilityList) {
    const container = document.getElementById('suitability-cards');
    if (!container) return;
    container.innerHTML = '';

    if (!suitabilityList || suitabilityList.length === 0) return;

    suitabilityList.forEach((suit, idx) => {
        const card = document.createElement('div');
        card.className = 'suitability-card';
        card.setAttribute('data-verdict', suit.verdict);

        const fillClass = `fill-${suit.verdict}`;
        const shouldPlant = suit.should_plant;
        const verdictBadge = shouldPlant
            ? `<span class="suit-verdict suit-verdict-yes">✅ Yes — Plant this crop</span>`
            : `<span class="suit-verdict suit-verdict-no">⚠️ Not recommended for planting</span>`;

        // Feature details
        let featuresHTML = '';
        if (suit.features && suit.features.length > 0) {
            const featureItems = suit.features.map(f => {
                const statusClass = `status-${f.status.toLowerCase()}`;
                return `
                    <div class="suit-feature">
                        <div>
                            <div class="suit-feature-name">${f.name}</div>
                            <div class="suit-feature-detail">
                                Yours: ${f.your_value} ${f.unit} · Ideal: ${f.ideal_min}–${f.ideal_max} ${f.unit}
                            </div>
                        </div>
                        <span class="suit-feature-status ${statusClass}">${f.status}</span>
                    </div>
                `;
            }).join('');

            featuresHTML = `
                <button class="suit-toggle-btn" onclick="toggleSuitDetails(this)" aria-expanded="false">
                    ▶ Show Feature Details
                </button>
                <div class="suit-features suit-details-hidden">
                    ${featureItems}
                </div>
            `;
        }

        // Tips
        let tipsHTML = '';
        if (suit.tips && suit.tips.length > 0) {
            tipsHTML = `
                <div class="suit-tips">
                    ${suit.tips.map(t => `<div class="suit-tip">${t}</div>`).join('')}
                </div>
            `;
        }

        card.innerHTML = `
            <div class="suit-header">
                <div class="suit-emoji">${suit.emoji || '🌱'}</div>
                <div class="suit-header-info">
                    <div class="suit-crop-name">${suit.crop}</div>
                    <div class="suit-match-row">
                        <div class="suit-match-bar-bg">
                            <div class="suit-match-bar-fill ${fillClass}" style="width: 0%;"></div>
                        </div>
                        <span class="suit-match-pct">${suit.overall_match}%</span>
                    </div>
                </div>
            </div>
            ${verdictBadge}
            <div class="suit-verdict-text">${suit.verdict_text || ''}</div>
            ${featuresHTML}
            ${tipsHTML}
        `;

        container.appendChild(card);

        // Animate the match bar
        const bar = card.querySelector('.suit-match-bar-fill');
        if (bar) {
            setTimeout(() => {
                bar.style.width = `${suit.overall_match}%`;
            }, 100 + idx * 150);
        }
    });
}


// Toggle feature details in suitability cards
function toggleSuitDetails(btn) {
    const details = btn.nextElementSibling;
    if (!details) return;

    const isHidden = details.classList.contains('suit-details-hidden');
    details.classList.toggle('suit-details-hidden');
    btn.setAttribute('aria-expanded', isHidden);
    btn.textContent = isHidden ? '▼ Hide Feature Details' : '▶ Show Feature Details';
}


function renderProbChart(allProbs, topCrop) {
    const chart = document.getElementById('prob-chart');
    chart.innerHTML = '';

    // Sort descending by probability
    const sorted = Object.entries(allProbs)
        .sort((a, b) => b[1] - a[1]);

    // Find max for scaling
    const maxProb = sorted[0]?.[1] || 1;

    sorted.forEach(([crop, prob], idx) => {
        const isTop = crop === topCrop;
        const barWidth = maxProb > 0 ? (prob / maxProb) * 100 : 0;

        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <span class="prob-crop-name">${crop}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill ${isTop ? 'highlight' : ''}" style="width: 0%;"></div>
            </div>
            <span class="prob-value">${prob.toFixed(1)}%</span>`;
        chart.appendChild(row);

        // Animate bars staggered
        const fill = row.querySelector('.prob-bar-fill');
        setTimeout(() => {
            fill.style.width = `${barWidth}%`;
        }, 50 + idx * 30);
    });
}


// ═══════════════════════════════════════════════════════
// Session History (localStorage)
// ═══════════════════════════════════════════════════════

function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch {
        return [];
    }
}

function saveHistory(history) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function addToHistory(inputValues, result) {
    const history = getHistory();
    history.unshift({
        timestamp: new Date().toISOString(),
        inputs: inputValues,
        crop: result.recommended_crop,
        confidence: result.confidence,
        emoji: result.emoji,
    });
    // Keep last 25
    if (history.length > 25) history.length = 25;
    saveHistory(history);
    renderHistory();
}

function renderHistory() {
    const history = getHistory();
    historyCount.textContent = history.length > 0 ? `(${history.length})` : '';

    if (history.length === 0) {
        historyList.innerHTML = '<li class="history-empty">🌱 No predictions yet. Make your first recommendation above!</li>';
        return;
    }

    historyList.innerHTML = history.map(h => {
        const t = new Date(h.timestamp);
        const timeStr = t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const dateStr = t.toLocaleDateString([], { month: 'short', day: 'numeric' });
        const inputStr = `N=${h.inputs.N} P=${h.inputs.P} K=${h.inputs.K}`;
        return `
            <li class="history-item">
                <span class="history-emoji">${h.emoji}</span>
                <div>
                    <div class="history-crop">${h.crop}</div>
                    <div class="history-inputs">${inputStr} · ${dateStr} ${timeStr}</div>
                </div>
                <span class="history-conf">${h.confidence}%</span>
            </li>`;
    }).join('');
}


// ═══════════════════════════════════════════════════════
// UI State Helpers
// ═══════════════════════════════════════════════════════

function setLoading(loading) {
    if (loading) {
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
    } else {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }
}

function showError(message) {
    errorBanner.textContent = `❌ ${message}`;
    errorBanner.classList.add('visible');
    resultsSection.style.display = 'none';
}

function hideError() {
    errorBanner.classList.remove('visible');
}

function resetForm() {
    form.reset();
    FIELD_MAP.forEach(f => {
        const el = document.getElementById(f.id);
        if (el) {
            el.classList.remove('error');
            el.value = '';
        }
        const errEl = document.getElementById(`error-${f.id.replace('input-', '')}`);
        if (errEl) errEl.textContent = '';
        updateRangeIndicator(f);
    });
    scenarioSelect.value = '';
    resultsSection.style.display = 'none';
    hideError();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}


// ═══════════════════════════════════════════════════════
// Form Submission
// ═══════════════════════════════════════════════════════

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();

    // Client-side validation
    if (!validateForm()) return;

    // Gather values
    const payload = {};
    FIELD_MAP.forEach(f => {
        payload[f.key] = parseFloat(document.getElementById(f.id).value);
    });

    setLoading(true);

    try {
        const result = await fetchPrediction(payload);
        renderResults(result, payload);
        addToHistory(payload, result);
    } catch (err) {
        showError(err.message || 'Something went wrong. Please try again.');
    } finally {
        setLoading(false);
    }
});


// ═══════════════════════════════════════════════════════
// New Prediction Button
// ═══════════════════════════════════════════════════════

if (newPredBtn) {
    newPredBtn.addEventListener('click', resetForm);
}


// ═══════════════════════════════════════════════════════
// History Toggle
// ═══════════════════════════════════════════════════════

historyToggle.addEventListener('click', () => {
    const isOpen = historyContent.style.display !== 'none';
    historyContent.style.display = isOpen ? 'none' : '';
    historyToggle.classList.toggle('open', !isOpen);
    historyToggle.setAttribute('aria-expanded', !isOpen);
});

historyClear.addEventListener('click', () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
});


// ═══════════════════════════════════════════════════════
// Mobile Navigation
// ═══════════════════════════════════════════════════════

if (hamburger) {
    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('mobile-open');
        hamburger.setAttribute('aria-expanded',
            navLinks.classList.contains('mobile-open'));
    });

    // Close mobile nav when a link is clicked
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('mobile-open');
            hamburger.setAttribute('aria-expanded', 'false');
        });
    });
}


// ═══════════════════════════════════════════════════════
// Initialise
// ═══════════════════════════════════════════════════════

renderHistory();

// Set initial range indicators
FIELD_MAP.forEach(f => updateRangeIndicator(f));
