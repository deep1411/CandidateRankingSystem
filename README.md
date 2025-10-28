# CandidateRankingSystem

> ML-powered pipeline to **score, rank, and re-rank** job candidates for specific roles (e.g., *"Aspiring Human Resources"*), with a **human-in-the-loop** feedback loop that learns from starred candidates.

---

## Overview

Hiring teams spend significant time identifying promising candidates. This project automates the **fitness scoring** and **ranking** of candidates from semi-structured sourcing data, then supports **interactive re-ranking** when a reviewer *stars* a preferred candidate. The starred feedback becomes a supervisory signal to refine future rankings.

**Core capabilities**

- Ingest candidate metadata (e.g., `job_title`, `location`, `connections`) and create a **fitness score** (0–1).
- Produce **ranked lists** by role-specific keywords (e.g., *"aspiring human resources"*).
- **Re-rank** after a reviewer *stars* a candidate (feedback loop).
- Provide **notebooks** for EDA, modeling, evaluation, and inference/demos.
- Encourage **bias awareness** and reproducibility (seeds, config blocks).

Last updated: *October 28, 2025*

---

## Data

The dataset is anonymized and uses a unique identifier per candidate. Minimal schema:

| Column       | Type   | Description                                                      |
|--------------|--------|------------------------------------------------------------------|
| `id`         | int    | Unique candidate identifier                                      |
| `job_title`  | text   | Candidate’s job title                                            |
| `location`   | text   | Candidate’s location                                             |
| `connections`| text   | Connection count (e.g., `500+` for over 500)                     |
| `fit`        | float  | Target variable: fitness score (0–1). Optional at training time. |

> Keywords for initial role matching include **"Aspiring human resources"** / **"seeking human resources"**. You can adapt keywords per role.

**Privacy note:** All direct personal identifiers have been removed. IDs are synthetic.

---

## Repository Structure

```
.
├── data/
│   ├── potential-talents_aspiring-hr.csv           # example source (anonymized)
│   └── ExtendedPotentialTalents.csv                # extended dataset (if available)
├── notebooks/
│   ├── PotentialTalents_Annotated.ipynb            # end-to-end EDA → modeling → ranking
│   ├── GPT5nano_Annotated.ipynb                    # auxiliary modeling utilities
│   ├── Llama_Annotated.ipynb                       # LLM finetune/inference scaffolding
│   └── Qwen_Annotated.ipynb                        # LLM chat template + training
├── src/
│   ├── data.py                                     # loading/cleaning helpers
│   ├── features.py                                 # text/categorical pipelines (TF-IDF/OHE/Scaling)
│   ├── model.py                                    # fit/predict, calibration, persistence
│   ├── ranking.py                                  # scoring → ranking, NDCG/MAP utilities
│   ├── feedback.py                                 # starred feedback → re-rank strategies
│   └── eval.py                                     # metrics & reports
├── README.md
└── LICENSE
```

> If your local layout differs, keep this structure as a *reference*. The notebooks work even if you only have `data/` and `notebooks/`.

---

## Quickstart

### 1) Environment

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -U pip wheel

# Suggested libs (trim as needed)
pip install numpy pandas scikit-learn scipy matplotlib tqdm             jupyter nbformat             xgboost lightgbm             shap
```

### 2) Data

Place CSVs under `data/` (filenames can vary). Columns should include `id`, `job_title`, `location`, `connections` and optionally `fit` if supervised labels are available.

### 3) Run the main notebook

Open in Jupyter and run top-to-bottom:

- `notebooks/PotentialTalents_Annotated.ipynb`

This notebook walks through:

1. **EDA** (schema checks, missingness, distributions)  
2. **Preprocessing** (text normalization, categorical handling, connection bucketing)  
3. **Feature engineering** (e.g., TF-IDF for titles + one-hot for locations + connection bands)  
4. **Modeling** (baseline Logistic Regression, Random Forest/GBMs)  
5. **Evaluation** (ROC-AUC/PR-AUC, **ranking metrics** like **NDCG@k**, **MAP@k**)  
6. **Initial Ranking** (sort by predicted fitness)  
7. **Re-ranking with Starred Feedback** (see below)  
8. **Export** (ranked CSVs/artifacts)

---

## Modeling & Ranking

### Baselines
- **Logistic Regression** over TF-IDF + categorical/numeric features (fast, transparent).
- **Tree Ensembles** (LightGBM/XGBoost/Random Forest) for non-linear interactions.
- Optional **Calibration** (Platt/Isotonic) to improve probability quality.

### Metrics
- **Classification**: ROC-AUC, PR-AUC, F1 (if using a decision threshold).
- **Ranking**: **NDCG@k**, **MAP@k**, **Recall@k**—primary success metrics for shortlist quality.

### Re-ranking with Starred Feedback

When a reviewer stars the *ideal* candidate within a result list, we update the ranking to reflect that preference.

Two simple, effective strategies:

1. **Ideal Profile Vector (Text Similarity)**  
   - Build/maintain a vector (e.g., TF-IDF centroid) of starred candidates for a given role.  
   - Re-score candidates by **cosine similarity** to this profile and blend with model score:  
     \( \text{final} = \alpha\cdot \text{model\_score} + (1-\alpha)\cdot \text{similarity} \)  
   - Update profile incrementally as more candidates are starred.

2. **Pairwise Preference Learning**  
   - Generate pairs (starred > non-starred) and train a lightweight **Logistic Regression** on feature **differences**.  
   - At inference, compute a pairwise score and combine with baseline probability for final ranking.

> Both methods improve with each star action and can be logged to show gains in NDCG/MAP over time.

---

## Command-line Sketch (optional)

If you add simple CLI wrappers under `src/`, you might expose flows like:

```bash
# Train and export artifacts
python -m src.model --train --data data/ExtendedPotentialTalents.csv --out artifacts/

# Score and rank for a specific role keyword
python -m src.ranking --score --data data/potential-talents_aspiring-hr.csv --role "aspiring human resources"     --model artifacts/model.pkl --out outputs/ranked_candidates.csv

# Apply a starred feedback update (by candidate id)
python -m src.feedback --star --id 12345 --role "aspiring human resources"     --profiles outputs/role_profiles.json --out outputs/reranked_candidates.csv
```

---

## Notebooks

- **PotentialTalents_Annotated.ipynb** – End-to-end pipeline with clear sections and next-steps.  
- **GPT5nano_Annotated.ipynb** – Auxiliary modeling utilities / experiments.  
- **Llama_Annotated.ipynb** – Llama-family finetuning or inference scaffolding.  
- **Qwen_Annotated.ipynb** – Qwen chat templates via `apply_chat_template`, training and inference examples.

> Export to HTML/PDF for sharing with non-technical stakeholders.

---

## Bias Awareness & Cutoffs

- Inspect score distributions across **location** or **connection** bands to avoid unfair thresholds.  
- Prefer **rank-based** shortlists (top-*k*, top-*p*) to hard probability cutoffs; tune *k* by reviewer capacity.  
- Keep an **audit log** of starred actions and resulting ranking changes.

---

## Roadmap

- [ ] Add a simple API/Gradio UI for search → rank → star → re-rank.  
- [ ] Online learning for pairwise preference model.  
- [ ] Automated bias reports and fairness dashboards.  
- [ ] Model registry + experiment tracking (MLflow/W&B).

---

## Contributing

Contributions are welcome! Please open an issue or PR:
1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-change`)
3. Commit with context (`feat: add NDCG@k metric`)
4. Open a PR and describe your change

---

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

## Acknowledgments

- Thanks to open-source ML libraries and the broader community.
- This project originated from the need to automate and improve candidate discovery and ranking while keeping humans in the loop.
