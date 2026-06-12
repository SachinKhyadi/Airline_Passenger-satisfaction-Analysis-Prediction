# Airline Passenger Satisfaction & Revenue Strategy Analysis
**R | Decision Tree | Neural Network | 103,904 records | 91.2% Accuracy**

> Built to answer one commercial question: *which service investments produce the highest satisfaction lift and for which passenger segments to maximize revenue?*

---

## Business Impact Summary

| Finding | Revenue Implication |
|---|---|
| Online boarding satisfaction predicts 23% higher overall satisfaction | Priority loyalty program targeting opportunity |
| Business class passengers weight inflight wifi 2.1× higher than economy | Premium service investment ROI signal |
| Personal travelers show highest sensitivity to service touchpoints | Segmentation-driven outreach and upsell strategy |
| Loyal customers show 31% higher baseline satisfaction than first-time flyers | Retention spend yields compounding revenue return |

---

## The Commercial Problem

Airlines invest across dozens of service categories like wifi, boarding, food, seating, entertainment with limited visibility into *which* investments actually move satisfaction, and *for whom*. Generic satisfaction improvements spread budget too thin. This analysis identifies the high-leverage levers by segment, enabling targeted, revenue-justified service investment.

---

## Dataset

- **103,904 passenger records** (train + test split)
- **22 service and demographic features**: inflight wifi, online boarding, food & drink, seat comfort, inflight entertainment, cleanliness, baggage handling, and more
- **Segments analyzed**: Loyal vs. first-time customers; Business, Eco, Eco Plus class; Personal vs. business travel purpose
- **Target**: Satisfied vs. Neutral/Dissatisfied

---

## Key Findings by Segment

### Segment 1 - Business Travelers (highest revenue per seat)
- **Top satisfaction driver**: Inflight wifi service (highest feature weight)
- **Implication**: Wifi reliability improvements have outsized ROI for the highest-yield segment; pricing premium wifi as a loyalty perk rather than a surcharge could improve both satisfaction and repeat booking

### Segment 2 - Personal Travelers (highest churn risk)
- **Top satisfaction driver**: Online boarding experience, then food & drink
- **Implication**: Digital touchpoints matter more than in-cabin service for this group — a counterintuitive finding that challenges typical service investment assumptions

### Segment 3 - Loyal Customers (retention value)
- **Baseline**: 31% higher satisfaction than first-time flyers
- **Implication**: Satisfaction investments protect existing revenue more than they generate new revenue; loyalty program enhancements should be framed as retention spend, not acquisition spend

### Segment 4 - First-Time Flyers (conversion opportunity)
- **Gap**: Significantly lower satisfaction across nearly all dimensions
- **Implication**: First-flight experience quality is disproportionately important to long-run revenue; targeted onboarding improvements could shift these passengers toward the loyal customer cohort

---

## Model Performance

| Metric | Decision Tree | Neural Network |
|---|---|---|
| Accuracy | **91.2%** | Comparable |
| Sensitivity | 88.99% | — |
| Specificity | 94.12% | — |
| Kappa Score | 0.8237 | — |
| Balanced Accuracy | 91.55% | — |

Both models were validated on held-out test data. Decision Tree chosen as primary for **interpretability** satisfaction drivers can be explained directly to non-technical stakeholders and leadership, which is a requirement in commercial analytics contexts.

---

## Top Satisfaction Drivers (ranked by model feature importance)

1. **Online boarding** - strongest single predictor across all segments
2. **Inflight wifi service** - strongest for business class; weak for economy
3. **Type of travel** - personal vs. business purpose predicts satisfaction independently of service ratings
4. **Class** — business class passengers report higher satisfaction even on equivalent service ratings
5. **Inflight entertainment** - significant for longer flights; less relevant for short-haul
6. **Food & drink** - second-ranked for personal travelers
7. **Seat comfort** - consistent predictor across all segments

---

## Methodology

**EDA**: Pie charts by customer type, gender, class, travel type; bar charts of mean service ratings by satisfaction class; boxplots of age and flight distance; correlation heatmap confirming multicollinearity of delay variables (Arrival Delay dropped).

**Decision Tree**: `rpart` with cross-validation and cost-complexity pruning for generalizability; `confusionMatrix` evaluation.

**Neural Network**: 2-layer architecture via `neuralnet`; numerical normalization and categorical encoding; comparable accuracy to Decision Tree with higher computational cost and lower interpretability.

---

## Tools

- **R**: `rpart`, `caret`, `neuralnet`, `ggplot2`
- Data cleaning, feature engineering, model training, cross-validation, evaluation

---

## Repository Contents

| File | Description |
|---|---|
| `Passenger_Satisfaction_Project.R` | Full analysis: EDA, modeling, evaluation |
| `airline_passenger_train.csv` | Training data (103,904 records) |
| `airline_passenger_test.csv` | Held-out test data |
| `BI.006.17.Final-Report.airlinepassenger.docx` | Full academic write-up |

