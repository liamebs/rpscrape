
# RPR and OR Analysis Report

## Overview

This report details the behavior and modeling suitability of two commonly used horse racing features:

- **RPR**: Racing Post Rating  
- **OR**: Official Rating

These features were analyzed across pre-race (`future scrape`) and post-race (`historical scrape`) datasets from March 24–25, 2025, to determine whether they introduce **data leakage** when used in predictive models.

---

## RPR (Racing Post Rating)

### What is RPR?

Traditionally, RPR is viewed as a **post-race performance score** assigned by Racing Post analysts. It reflects how well a horse ran in a given race.

However, based on new information from a trusted Racing Post forum contributor:

> *“Racing Post Ratings are adjusted/fiddled for each race at the overnight stage to take into a number of factors, including trip/ground conditions and trainer form... to ensure that the horses considered by the handicapper to have the best chances of running well have the highest adjusted ratings.”*

### Updated Interpretation

RPR is not strictly a post-race metric. It is **dynamically adjusted overnight**, reflecting the Racing Post's expert projection of a horse’s suitability for today's conditions.

It serves more as a **domain-informed prior** — akin to a "form rating with expert context" — rather than a purely retrospective score.

### Observations in This Project

- RPR was present in the pre-race scrapes for many runners.
- RPR was often missing in the post-race scrapes, indicating that it is not always updated promptly post-race.
- Horses with no prior runs or too few data points often had missing RPRs.

### Strategic Insight

If the poster is correct, **RPR is adjusted during the night before the race** to reflect expert opinion. This means:

- It is *not* derived from the current race's result
- It can be safely scraped on **the morning of the race day**
- Post-race updates still occur, but only **after** some delay

### Risk Assessment

| Criterion                  | Assessment       |
|---------------------------|------------------|
| Present pre-race          | ✅ Yes            |
| Adjusted overnight        | ✅ Yes            |
| Based on current race outcomes | ❌ No         |
| Leak risk                 | ⚠️ Low            |
| Recommendation            | ✅ Use with care  |

### Recommendation

You **can use RPR** as a feature in your model, **provided you scrape it on race day, before races begin**. This ensures you capture the "overnight expert adjustment" without risking post-race leakage.

RPR is best treated as a **prior signal** — a Racing Post-informed expert score estimating how well the horse fits the current race.

---

## OR (Official Rating)

### What is OR?

OR is an official handicap mark assigned by racing authorities to quantify a horse’s ability. It is publicly published and **included in racecards pre-race**.

### Observations in This Project

- OR was present in both pre- and post-race data.
- OR changed in only **0.87% of cases**, showing strong stability.

### Risk Assessment

| Criterion                  | Assessment     |
|---------------------------|----------------|
| Present pre-race          | ✅ Yes          |
| Stable post-race          | ✅ Yes          |
| Leak risk                 | ❌ None         |
| Recommendation            | ✅ Safe to use  |

---

## Summary Table

| Feature | Present Pre-Race | Adjusted for Conditions | Changes Post-Race | Leakage Risk | Recommended Use |
|--------|------------------|-------------------------|-------------------|---------------|-----------------|
| RPR    | ✅ Yes            | ✅ Overnight             | ✅ Sometimes       | ⚠️ Low         | ✅ Yes (if scraped early) |
| OR     | ✅ Yes            | ❌ No                    | ❌ Rarely          | ✅ None        | ✅ Yes           |

---

## Conclusion

- ✅ **OR is a reliable, pre-race feature** and is safe for model inclusion.
- ✅ **RPR is an expert-adjusted overnight signal** that can be used in modeling, *if scraped responsibly*.
- 📌 To use RPR without risk, **scrape it on race day before races begin**, and never include values updated after race completion.

This makes RPR a valuable expert-derived signal — and not a leak — when handled with care.


---

## Appendix: The Nature of Ratings — A Philosophical Perspective

A valuable contribution from a Racing Post forum poster offers a deeper view of what ratings truly are:

> *“All ratings are made up... A rating is merely an opinion of a horse’s ability based on the form information available to a compiler and different compilers will come up with different interpretations of the same data.”*

This includes:
- Racing Post Ratings (RPR)
- Official Ratings (OR)
- Timeform figures
- Your own machine learning model

### 🧠 What This Means for Modeling

Ratings are not facts — they are **human or institutional models** that:
- Interpret past data
- Express informed guesses
- Encode domain-specific heuristics

They are **structured subjectivity**.

### ✅ RPR and OR Are Inputs, Not Truth

Rather than treating ratings as leaks or fixed truths, we can treat them as:
- Public priors
- Expert features
- Opinionated baselines

This is not a flaw. It’s an **opportunity to model consensus vs reality**, and calibrate your system accordingly.

### 📌 Final Takeaway

> You are not leaking the target by using RPR or OR.  
> You are incorporating **domain knowledge** from external models, and giving your own model the chance to agree or disagree.

This makes RPR and OR **not liabilities**, but valuable context when handled carefully.


---

## Appendix: Confirmation of RPR Post-Race Mutation

Following visual inspection of runners from Wincanton on March 24, 2025, it was confirmed that:

> **RPR values scraped before the race differ from those later shown in historical datasets.**

### 🔍 Implications

- RPR is **not immutable** — it may be edited after the race has run.
- Even though RPR is available before the race, **the version found in historical scrapes is not necessarily the same**.
- This validates earlier forum insights: **RPR is provisional pre-race**, and **can be revised** based on new form lines or analyst review.

### ⚠️ Modeling Consequence

| Use Case | Action |
|----------|--------|
| Use RPR from race-day scrape | ✅ Safe |
| Use RPR from historical data | ❌ Unsafe — may be post-edited |
| Train model on archived pre-race RPRs | ✅ Valid |
| Recreate RPR from backfilled data | ❌ Risk of leakage |

### ✅ Final Recommendation

> Archive and tag pre-race RPRs on the morning of race day.  
> These are the only versions you can trust to reflect what the public and experts saw before the outcome was known.
