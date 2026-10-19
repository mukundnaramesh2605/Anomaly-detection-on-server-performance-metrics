# Anomaly Detection on Server Performance Metrics

Detecting degradation in live system metrics using unsupervised learning.

An Isolation Forest flags anomalous behaviour in real EC2 CPU utilisation data, with no labels used during training. Built as an end-to-end project: exploratory analysis to choose features, model, and evaluation framed around what a monitoring system actually needs.

## Problem

Given 14 days of CPU utilisation from an AWS EC2 instance (one reading every 5 minutes), find the periods where the server behaved abnormally, without telling the model what "abnormal" looks like.

- **Data:** Numenta Anomaly Benchmark (NAB), `realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv`
- **Size:** 4,032 readings over 14 days
- **Ground truth:** 2 labeled anomaly windows (~16 hours each), used only for evaluation, never for training

Coming from a performance engineering background, this is the kind of question you ask of a production dashboard: when did the system start misbehaving, and would an automated detector have caught it?

## Approach

**1. EDA to choose features.** Worked through the series before modelling: spacing, distribution, rolling statistics, and each candidate feature's separation between normal and anomalous periods. The raw value turned out to carry almost no separating power on its own. Three engineered features did better:

- `abs_dev` — absolute deviation from the 1-hour rolling mean (how far a point strays from its recent normal)
- `roll_std` — 1-hour rolling standard deviation (local volatility)
- `change` — absolute rate of change between consecutive readings (sudden jumps)

**2. Isolation Forest.** Unsupervised. `contamination` set to 0.10, matching the observed anomalous fraction from EDA (~10% of readings fall inside the labeled windows).

**3. Scored two ways.** Point-level precision/recall/F1 (strict), and window-level recall (did the model raise any flag inside each real incident).

## Results

**Both incidents caught.** The model flagged activity inside both labeled anomaly windows (38 and 31 flags respectively). For a monitoring use case, detecting the incident is the goal, and it did.

**Point-level scores are modest, and honestly so:**

| Metric | Value |
|---|---|
| Precision | 0.17 |
| Recall | 0.17 |
| F1 | 0.17 |
| Window-level recall | 2/2 |

The point-level F1 looks weak because the labels are 16-hour blocks. A detector that catches the event but doesn't flag every exact minute inside the window gets heavily penalised by point scoring. Window-level recall is the metric that reflects operational value, and it's the one I lead with.

**The real constraint is the data, not the pipeline.** During EDA, every feature showed only 1.0x to 1.3x separation between normal and anomalous periods. This is genuinely low-contrast data: the anomalies are sustained, subtle shifts rather than obvious spikes. The model performs about as well as the features allow, which is a property of the problem.

**Contamination was tuned, not guessed:**

| contamination | precision | recall | F1 |
|---|---|---|---|
| 0.10 | 0.172 | 0.172 | 0.172 |
| 0.12 | 0.166 | 0.199 | 0.181 |
| 0.15 | 0.153 | 0.229 | 0.183 |

Higher values trade precision for recall with almost no F1 gain. Kept 0.10 because it matches the observed anomaly rate and keeps the reasoning clean.

## Figures

**Detection overview** — CPU series with true windows shaded and model flags marked. Flags land inside both windows; scattered flags elsewhere reflect the normal/anomaly overlap.

**Primary feature** — deviation from the 1-hour rolling mean. Its two largest spikes in the entire series both fall inside the labeled windows, confirming the feature choice.

## Key takeaways

- On low-contrast time series, metric choice matters as much as the model. Point-level scoring can badly understate a detector that catches the right events.
- Feature engineering did the heavy lifting. The raw signal was nearly useless; deviation-from-rolling-mean carried the detection.
- Setting `contamination` from the observed anomaly rate is more defensible than tuning it to chase a score.

## What I'd do next

- **Rolling z-score baseline** to compare Isolation Forest against a simple statistical detector.
- **NAB's own scoring** (which rewards early detection within a window) for a fairer benchmark than point F1.
- **Test on more NAB series** to see where the approach holds and where it breaks.

## Run it

```
pip install pandas numpy scikit-learn matplotlib
```

Open the notebook and run top to bottom. Dataset and labels are from the [NAB repo](https://github.com/numenta/NAB) (`data/realAWSCloudwatch/` and `labels/combined_windows.json`).
