# Cloud Predict — Master Technical Architecture (v2)

## 1. Project Overview

Cloud Predict is an open-source, AI-driven hybrid-cloud storage tiering orchestrator — a lightweight, on-premise alternative to proprietary systems like AWS S3 Intelligent-Tiering.

Rather than static, rule-based policies ("move if untouched for 30 days"), the system frames tiering as a financial decision problem: for every file, it weighs the storage cost saved by moving it to cold storage against the expected penalty of retrieving it back if that turns out to be the wrong call. A machine learning model estimates the probability of that "wrong call," and an explicit cost formula (Section 3, Component D) turns that probability into a dollar-denominated migrate/keep decision.

## 2. Tech Stack & Development Constraints

- **Backend:** Python 3.x, FastAPI
- **Storage:** MinIO (local S3-compatible object storage)
- **ML:** Scikit-Learn (preprocessing, pipelines, baseline models, metrics) + XGBoost (production classifier)
- **Data handling:** Pandas, NumPy
- **Scheduling:** APScheduler — in-process, no external broker, keeps the stack lightweight
- **Config:** a single `config.yaml` (or `pydantic-settings`) holding tunable pricing and thresholds — see Component F
- **Dev environment:** Docker Compose bringing up MinIO + the FastAPI app together, for a reproducible local setup
- **Testing:** pytest, with MinIO mocked via a test bucket or `moto`

**Hard constraint, unchanged:** no React, Node.js, or npm-based frontend of any kind. Any UI is vanilla HTML/CSS/JS served via FastAPI + Jinja2, or pure REST endpoints. A CDN-hosted chart library (e.g. Chart.js via a plain `<script>` tag, no build step) is fine for a dashboard and doesn't violate this.

## 3. Core System Components

### Component A — The Storage Layer (MinIO)

Unchanged: two buckets, `hot-tier-bucket` (fast/expensive) and `cold-tier-bucket` (slow/cheap).

*Addition:* uploads and downloads should go through MinIO **presigned URLs** rather than proxying file bytes through FastAPI — see Component C.

### Component B — The Data Simulation Module

Purpose unchanged: generate realistic synthetic workloads for ML training using a Zipfian (long-tail) access-frequency distribution — a good, realistic modeling choice that matches how real storage/cache access patterns behave.

Two problems in the original spec, fixed here:

1. **It needs to simulate time, not produce a static table.** A one-shot snapshot where `Access_Frequency` is computed from a file's entire history can't produce valid training labels (see Component D). Instead, simulate a rolling clock: generate a stream of timestamped access *events* per file over a simulated window (Zipfian-weighted so popular files get more events), and derive features **as of** any chosen decision point using only the events before it.
2. **Leakage risk.** If a feature is computed using events that happen *after* the point you're predicting from, the model will look artificially accurate in testing and fail in production. Every feature must be computed strictly from data available before the decision timestamp.

**Output:** an event log — `File_ID, Event_Timestamp, Event_Type [upload / access]` — plus a static metadata table — `File_ID, File_Size_MB, Creation_Date, File_Type`. The event log should use the same schema as the real production logger in Component C, so simulated and real data feed the same downstream training pipeline without translation.

### Component C — The FastAPI Control Layer

- **Upload/Download API** — issues MinIO presigned URLs for the client to use directly, and logs the metadata event once the transfer completes, instead of buffering file bytes through the Python process.
- **Analytics API** — reports current tier distribution and cost savings, computed against the *same* pricing config used by the ML Orchestrator (Component D/F), so the number shown in a dashboard can never drift out of sync with what the model is actually optimizing for. Report savings against a naive "everything in hot tier" baseline so the ML value-add is demonstrable.
- **Trigger API** — the original spec has an internal contradiction: this endpoint is described as "manual," while the execution flow (Section 4) says the model runs "at a scheduled interval." Resolution: run APScheduler in-process for automatic evaluation+migration cycles on a configurable interval, **and** keep this endpoint so an operator can force an immediate cycle on demand (useful for demos and testing). Both coexist; neither replaces the other.

Two gaps the original plan doesn't mention at all, worth at least a minimal pass:
- **Auth** — even a simple API-key header is worth having before this touches anything resembling real data.
- **Input validation** — file-size limits and basic type checks on upload (FastAPI's `UploadFile` plus a size cap is enough; no need to over-build this for a lightweight project).

### Component D — The ML Orchestrator (Cost-Penalty Engine)

This is where the original plan is vaguest, and it's the component that actually carries the project's "AI-driven" claim — so it gets the most precision here.

**What the model predicts.** For a file *f*, at decision time *t*, with a chosen horizon *H* (e.g. 14 days):

```
P_access(f, t, H) = probability that f is accessed at least once in (t, t+H]
```

Structurally, this is the same shape as churn/propensity prediction — recency + frequency features predicting a future binary event — so standard techniques for that problem family apply directly.

**Training labels (missing from the original plan entirely).** From the event log in Component B, sample decision points per file and look forward *H* days: label = 1 if an access event occurs in that window, else 0. Features (recency, frequency, size, type, age) are computed only from events before the decision point, per the leakage note in Component B. The train/test split must be **time-based** — train on earlier decision points, test on later ones — not a random shuffle, which would leak information across the boundary.

**Class imbalance.** Given the Zipfian distribution, most files won't be touched in any given window, so "will be accessed" is the rare class. Use `scale_pos_weight` (XGBoost's built-in imbalance handling) and evaluate with precision/recall/PR-AUC — raw accuracy will be misleading here.

**The actual decision rule (not specified anywhere in the original plan).** Don't threshold the classifier at 0.5 and act on a binary label — feed its raw probability directly into an expected-value calculation, so the real cost asymmetry (wrongly migrating a soon-to-be-accessed file is expensive; unnecessarily keeping one in hot tier is only mildly wasteful) drives the decision instead of an arbitrary cutoff:

```
Storage_Savings(f)   = Size_GB(f) × (Hot_$/GB/mo − Cold_$/GB/mo) × (H / 30)
Retrieval_Penalty(f) = Egress_$/GB × Size_GB(f) + Latency_Penalty_$        # both from config, Component F

EV_migrate(f) = (1 − P_access(f,t,H)) × Storage_Savings(f)  −  P_access(f,t,H) × Retrieval_Penalty(f)

migrate f → cold-tier-bucket   if EV_migrate(f) > 0
```

This one formula is what turns "financial risk evaluation" from a sentence in the original plan into something a coding tool can implement directly, and it's the piece most worth getting right before handing this off.

**Retraining, not "live weight updates."** The original plan describes the model's weights updating on every misprediction — imprecise for a tree ensemble, which doesn't do online updates the way a neural net would. Two real options:
- **Simple and robust:** log every access event, including cold-tier retrievals, to the shared event log; retrain from the full log on a schedule (weekly, or after N new events).
- **Middle ground:** XGBoost supports continuing training from a previous booster via the `xgb_model=` argument, appending new trees rather than retraining from scratch — closer to the original "update on misprediction" intent, implemented correctly.

### Component E — The Migration Engine

The original copy → verify → delete sequence is the right pattern — it avoids the data-loss risk of a naive move and should stay exactly as specified. Additions:

- **Failure handling:** if copy succeeds but the integrity check or delete fails, the retry must be safe — check whether the object already exists at the destination before re-copying, rather than blindly re-running the full sequence.
- **Integrity check:** compare ETags/checksums between source and destination objects; MinIO's SDK exposes these natively.
- **Batching:** throttle or batch large migration runs so one evaluation cycle doesn't saturate I/O.
- **Promotion, not just demotion.** The original plan only ever moves files hot → cold. Add the symmetric case: when a cold-tier file is retrieved, check the same `EV_migrate` formula in reverse and promote it back to `hot-tier-bucket` if it now favors hot storage. This is standard in real tiering systems and reuses the copy/verify/delete machinery this component already has.

### Component F — Pricing & Cost Configuration *(new)*

The original plan never states where `Hot_$/GB/mo`, `Cold_$/GB/mo`, or `Egress_$/GB` come from, even though both the Analytics API and the ML Orchestrator depend on them. Pull these into one explicit, editable config (`config.yaml`, or a small table) instead of hardcoding them. This is what keeps the dashboard's "cost savings" number consistent with what the model is actually optimizing for, and lets the project be demoed against different pricing assumptions (AWS list prices, on-prem hardware amortization, etc.) without touching code.

## 4. Execution Flow (Data Lifecycle)

1. **Ingestion** — client requests a presigned upload URL from FastAPI, uploads directly to `hot-tier-bucket`; FastAPI logs the event.
2. **Tracking** — every upload and access is appended to the shared event log (SQLite is fine at this scale — worth noting its single-writer limitation only if this ever needs to handle heavy concurrent writes, which is outside the project's stated lightweight scope).
3. **Evaluation** — on the APScheduler interval, or via a manual Trigger API call: recompute features for every hot-tier file as of "now," score them with the current model, and compute `EV_migrate` using the live pricing config.
4. **Migration (demotion)** — the Migration Engine moves every file with `EV_migrate > 0` to `cold-tier-bucket`.
5. **Retrieval & promotion** — a request for a cold-tier file is served transparently, logged as an access event, and checked against the reverse `EV_migrate` for possible promotion back to `hot-tier-bucket`.
6. **Retraining** — on its own schedule, using the full event log (Component D).

## 5. Non-Functional Notes

Kept deliberately light, in line with the project's lightweight constraint — worth being aware of, not worth over-building:

- **Security:** a simple API-key check on write endpoints; upload size/type validation.
- **Testing:** pytest around the FastAPI endpoints, and specifically around the `EV_migrate` formula — it's pure arithmetic with no I/O, so it's the cheapest and most valuable thing here to unit-test.
- **Dev environment:** a `docker-compose.yml` bringing up MinIO and the FastAPI app together is a small addition that pays for itself in setup time and fits the "on-prem" framing well.

## 6. Open Design Questions for Implementation

Worth deciding explicitly (or flagging to whatever coding tool picks this up):

- Horizon `H` for the access-prediction window — 14 days is a reasonable starting default.
- Retraining cadence — event-count-based vs. calendar-based.
- Whether promotion is checked on every retrieval, or only past a repeat-access threshold.
- Starting values for the pricing config — AWS S3 Standard vs. Glacier list prices work as an initial placeholder.
