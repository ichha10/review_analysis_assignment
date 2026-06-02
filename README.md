# 🏨 Hotel Curator — Attribute Extraction & Tiering Pipeline

A production-style ML pipeline that extracts structured quality signals from hotel review text and maps them to tier ratings (**Elite / Superior / Premium / Fail**) for use in corporate travel booking.

---

## Chosen Attributes

| Attribute | Type | Coverage |
|---|---|---|
| **Cleanliness** | Sentiment | 9,589 clauses |
| **Staff Service** | Sentiment | 24,364 clauses |
| **Wi-Fi Quality** | Sentiment | 2,271 clauses |
| **Noise Level** | Sentiment | 7,255 clauses |
| **Location Accessibility** | Factual + Sentiment | 17,457 clauses |

---

## Project Structure

```
hotel_assignmet/
├── README.md
├── WRITEUP.md
├── requirements.txt
├── tripadvisor_hotel_reviews.csv        # Raw dataset
├── data/
│   └── labeled_clauses.csv             # Weak supervision labels
├── models/
│   └── attribute_classifier/           # Saved model artifacts
├── src/
│   ├── preprocess.py                   # Clause splitting utilities
│   ├── labeler.py                      # Hybrid labeling pipeline
│   ├── model.py                        # Classifier wrapper
│   ├── aggregator.py                   # Tier mapping + evidence
│   └── api.py                          # FastAPI inference endpoint
├── ui/
│   └── index.html                      # Evidence visualization UI
├── tests/
│   ├── test_preprocess.py
│   ├── test_labeler.py
│   └── test_aggregator.py
├── 01_eda.ipynb                        # Data exploration
├── 02_labeling.ipynb                   # Labeling pipeline
├── 03_modeling.ipynb                   # Model training & evaluation
└── 04_aggregation_and_tiers.ipynb      # Tiering + evidence surfacing
```

---

## How to Run End-to-End

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order

```bash
jupyter notebook 01_eda.ipynb
jupyter notebook 02_labeling.ipynb
jupyter notebook 03_modeling.ipynb
jupyter notebook 04_aggregation_and_tiers.ipynb
```

### 3. (Optional) Start the inference API

```bash
uvicorn src.api:app --reload
# Visit: http://localhost:8000/hotels/{hotel_id}/attributes
```

### 4. (Optional) Open the evidence UI

Open `ui/index.html` in any browser.

---

## Pipeline Overview

```
Raw Reviews (20,491)
      ↓
Clause Splitting (comma/punctuation split)
      ↓ 203,729 clauses
Hybrid Labeling
  ├── Keyword Triggers  → which attribute?
  └── DistilBERT SST-2  → positive / negative?
      ↓ labeled_clauses.csv
Model Training (DistilBERT fine-tuned, per attribute)
      ↓
Aggregation & Tiering
  ├── signal_score = pos / (pos + neg)
  ├── Elite ≥ 0.80 | Superior 0.60–0.79 | Premium 0.40–0.59 | Fail < 0.40
  └── Low-evidence hotels → "Uncertain"
      ↓
Evidence Surfacing (top-3 sentences per hotel × attribute)
```

---

## Tier Definitions

| Tier | Signal Score | Meaning |
|---|---|---|
| **Elite** | ≥ 0.80 | Consistently excellent |
| **Superior** | 0.60–0.79 | Mostly good |
| **Premium** | 0.40–0.59 | Mixed / acceptable |
| **Fail** | < 0.40 | Predominantly negative |
| **Uncertain** | < 3 mentions | Not enough evidence |

---

## Requirements

See `requirements.txt` for full list. Key dependencies:
- `pandas`, `numpy` — data processing
- `transformers`, `torch` — DistilBERT model
- `scikit-learn` — evaluation metrics
- `fastapi`, `uvicorn` — inference API (optional)
