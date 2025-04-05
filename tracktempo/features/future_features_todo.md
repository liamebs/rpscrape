
# 🧠 Future Feature Additions: Beyond the Current Feature Extractor Stub

This document outlines **additional features** that are present in the JSON data or derivable from domain knowledge, but are **not yet included** in the `feature_extractor_stub.py`.

These features have **modeling potential**, but may require:
- Engineering or transformation
- Historical aggregation
- Text parsing or NLP
- External data sources

---

## 🧩 Structured But Untapped Features

### 🔁 `form` (Recent Form String)
- Format: e.g., `"123P0F"` (past race results)
- ✅ Potential: Encode as recent performance trend
- 🔧 Needs:
  - Decoding into position categories
  - Possibly mapped to numeric pattern (e.g., 1/0/2 = win/place/other)

---

### ⏱️ `days_since_wind_surgery`, `medical_history`
- 📈 Signals like wind operations, tongue ties, etc.
- ✅ Potential: Huge market signal
- 🔧 Needs:
  - Historical tracking of surgery events
  - Backdated parsing (timeline of treatments)

---

### 📊 Course/Distance/Going Win Records
- Fields like: `course_wins`, `trip_wins`, `going_wins`
- ✅ Potential: Surface, stamina, and familiarity indicators
- 🔧 Needs:
  - Aggregation over horse history
  - May be sparse or noisy

---

### 🧠 NLP: Commentary & Spotlight Text

#### 📝 `comment`, `spotlight`
- Source: Racing Post's expert write-ups
- ✅ Potential: Encode expert sentiment and context
- 🔧 Needs:
  - Text cleanup (remove boilerplate)
  - Sentiment extraction (positive/neutral/negative)
  - Embedding via NLP model (e.g., MiniLM, BERT)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
vector = model.encode(runner['spotlight'])
```

Can yield 384- or 768-dimensional embeddings to be appended to model input.

---

## 🛠️ Derived Composite Features (Engineered)

### 📐 Race-Relative Features
| Feature | Description |
|--------|-------------|
| `rpr_rank` | Runner's rank in race by RPR |
| `or_rank` | OR percentile in race |
| `draw_rank` | Relative draw advantage |

---

### 🏇 Trainer & Jockey Dynamic Forms
| Feature | Description |
|--------|-------------|
| `trainer_form_percentile` | Rank of trainer win% in race |
| `jockey_inform_score` | Recent win%, form streak, etc. |

---

### 📈 Risk Indicators
| Feature | Description |
|--------|-------------|
| `layoff_flag` | Long gap since last race |
| `form_volatility_score` | Pattern of placements (1–0–1–0) |
| `risk_index` | Composite feature from surgery, layoff, form, etc. |

---

## 🧱 Summary

These features are **high-potential and domain-aligned**, but are deferred for now due to:

- Time constraints
- Need for data pipelines
- NLP model integration or domain aggregation

Once your core model is stable, you can **revisit these areas** to layer in even deeper insight and signal.

