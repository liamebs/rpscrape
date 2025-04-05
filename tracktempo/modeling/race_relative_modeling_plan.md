
# Race-Relative Horse Racing Model: Plan of Action

## 🧭 Objective

Design and implement a race-aware machine learning pipeline for horse race prediction, where each runner is evaluated **not in isolation**, but **relative to the field** they compete in.

---

## 🔍 Why Relative Modeling?

Traditional modeling treats horses in absolute terms:  
> *“Does Horse A have the characteristics of a winner?”*

But in racing, outcomes are **inherently relative**:
> *“Is Horse A better than the other runners **in this race**?”*

To model this effectively, the approach must:
- **Normalize or rank features per race**
- Understand **field strength**
- Learn **intra-race dynamics**

---

## ✅ Phase 1: Build a Clean Dataset (0–4 Weeks)

**Goal:** Accumulate a critical mass of leak-free, pre-race data (e.g. 1,000+ runners)

### Key Actions:
- ✅ Continue daily scraping (pre-race only)
- ✅ Archive `rpr` with timestamps
- ✅ Validate and store in structured format (`race_id`, `runner_name`, `rpr`, `or`, etc.)
- ⏳ Wait until mid-April for 2–4 weeks of clean coverage

---

## ⚙️ Phase 2: Implement Relative Feature Engineering (Weeks 2–5)

**Goal:** Transform absolute features into race-relative context

### Key Transformations:
| Feature | Relative Transformations |
|---------|--------------------------|
| `rpr`   | Rank, percentile, difference from race mean |
| `or`    | Delta from top-rated, z-score in field |
| `draw`  | Normalized draw, distance from middle |
| `age`   | Age rank, age delta from average |
| `field_size` | Field-aware interactions (e.g. draw × field size) |

### Suggested Feature Functions:
- `rank_within_race(df, feature)`
- `zscore_within_race(df, feature)`
- `softmax_within_race(df, proba_column)`

---

## 🔮 Phase 3: Train Race-Relative Classifier (Weeks 4–6)

**Goal:** Train a model to predict each horse’s probability of winning *relative to its field*

### Approach:
- XGBoost with `binary:logistic` or `rank:pairwise`
- Use grouped training by `race_id` (GroupKFold or GroupShuffleSplit)
- Evaluate using:
  - Accuracy / Log Loss
  - **Softmax-normalized Log Loss**
  - Simulated ROI / Top-N hit rate

---

## 🧠 Phase 4: Advanced Architectures (Optional / Beyond 6 Weeks)

**Goal:** Explore structured modeling of the entire race as a unit

### Advanced Architectures:
- **Race as a set of runners** → use a **Set Transformer**
- **Graph-based relationships** (e.g. trainer, jockey, form lines)
- Full neural net pipeline that:
  - Inputs: matrix of runners × features
  - Outputs: probability vector for that race

---

## 📦 Model Outputs

Predictions should:
- Be softmax-normalized within each race
- Include:
  - Win probability
  - Rank within field
  - Margin over 2nd best

---

## 🧰 Tools & Infrastructure

- Language: Python (Pandas, XGBoost, Scikit-learn)
- Data storage: daily flat files → combined inference/training set
- Archival: timestamped race-day scrapes
- Versioning: include race scrape date in filename for traceability

---

## ✅ Final Takeaways

- 🧠 Racing is **relational** — models must reflect that
- 🔐 Pre-race scrapes are the **only trustworthy RPR source**
- 🎯 Your modeling pipeline should think **within races**, not across all runners
- 📆 With just 2–4 weeks of data, you’ll be ready to begin testing the architecture

---

Stay focused on **race context**, and you’ll be building not just a predictive model — but a handicapping system that actually understands racing.
