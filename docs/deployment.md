# Deployment guide

This project has two complementary deployment surfaces:

1. **GitHub Pages** — a static showcase built from `docs/`. It is public, fast, and requires no server.
2. **Streamlit** — the interactive workbench (`app.py`). It can run locally, in Docker, or on Streamlit Community Cloud.

The GitHub Pages site is not the model endpoint; it is documentation and a visual tour. The interactive app loads the serialized model bundle and does not retrain on page load.

## 1. GitHub Pages (static showcase)

### One-time repository setup

1. Open **GitHub → repository → Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Ensure the default branch is `master` (this repository keeps `master`).
4. Run the **Deploy showcase** workflow from the Actions tab, or push a change under `docs/`.

The workflow is [`.github/workflows/pages.yml`](../.github/workflows/pages.yml). It:

- uploads the `docs/` folder with `actions/upload-pages-artifact`;
- deploys it with `actions/deploy-pages`;
- publishes the site at `https://mlaharsh.github.io/Customer_Churn_Prediction/`.

### Verify

```bash
gh run list --workflow pages.yml --limit 5
gh run watch <run-id>
curl -I https://mlaharsh.github.io/Customer_Churn_Prediction/
```

Expected: HTTP 200 and a page titled **Churn Signal — Customer Churn Prediction**.

### Update the showcase

Edit `docs/index.html` and the images under `docs/assets/`. Push to `master`. The workflow path filter only runs when `docs/**` or the workflow file changes.

### Optional CLI setup

If Pages is not enabled yet, an authenticated owner can enable it through the API:

```bash
gh api repos/mlaharsh/Customer_Churn_Prediction/pages \
  -X POST \
  -f build_type=workflow
```

Then run the Pages workflow. If Pages already exists, use `-X PUT` instead of `-X POST`.

## 2. Streamlit Community Cloud

The app is designed to deploy without secrets or a database. The model, metrics, and evaluation CSVs are committed to the repository.

### Step by step

1. Push the repository to GitHub (branch `master`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Select **New app**.
4. Choose:
   - **Repository:** `mlaharsh/Customer_Churn_Prediction`
   - **Branch:** `master`
   - **Main file path:** `app.py`
   - **Python version:** 3.11 (recommended; 3.12 is also tested)
5. Deploy.

The app installs `requirements.txt`, loads `models/churn_model.joblib`, and starts on the Streamlit domain. No environment variables are required.

### Important status note

The repository includes a working local app and a verified CI smoke test. A public Streamlit Community Cloud URL is **not asserted in the README** unless a live link is added. If you deploy it, add the URL to the README table and the docs index so the claim stays honest.

### If deployment fails

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Model artifact not found` | `models/churn_model.joblib` is missing from the branch | Run `python -m churn.train` locally and commit `models/` |
| `joblib` unpickle warning / error | scikit-learn version mismatch | Use the version range in `requirements.txt`; retrain if necessary |
| App starts but data pages fail | `reports/evaluation/*.csv` missing | Re-run training and commit `reports/evaluation/` |
| Build times out during install | heavy dependency resolution | Pin the versions from the tested environment or use Docker |
| Batch upload rejected | missing columns or duplicate IDs | Use the template in the app; check the validation rules |

## 3. Docker

Build and run:

```bash
docker build -t churn-signal .
docker run --rm -p 8501:8501 churn-signal
```

Open http://localhost:8501.

The image uses Python 3.11 slim, installs `libgomp1` for scikit-learn, and includes a health check against `/_stcore/health`.

To pass a different port:

```bash
docker run --rm -p 9000:8501 churn-signal
```

## 4. Local deployment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m churn.train
streamlit run app.py
```

The app runs at http://localhost:8501.

## 5. Retraining and artifact updates

```bash
python -m churn.train                 # full 5-fold comparison + figures
python -m churn.train --cv-splits 2 --no-figures   # faster CI-style rebuild
```

Generated files:

```text
models/churn_model.joblib
models/feature_importance.csv
models/metrics.json
reports/evaluation/model_comparison.csv
reports/evaluation/test_predictions.csv
reports/evaluation/risk_deciles.csv
reports/evaluation/segment_*.csv
reports/evaluation/summary.md
reports/figures/*.png
```

Commit the model, metrics, and evaluation outputs together. The app and README depend on them. If you change the threshold policy, update the model card and the static showcase metrics as well.

## 6. CI

`.github/workflows/ci.yml` runs on pushes to `master`, pull requests, and manual dispatch:

- **test** — Python 3.11 and 3.12, compile check, pytest;
- **train** — two-fold rebuild and artifact verification;
- **app-smoke** — starts Streamlit and checks the health endpoint.

The Pages workflow is independent. A documentation-only change can deploy the showcase without rerunning training.

## 7. Operational checklist

- [ ] `python -m pytest -q` passes.
- [ ] `python -m churn.train` completes and regenerates artifacts.
- [ ] `streamlit run app.py` loads all five pages.
- [ ] Batch upload and template download work.
- [ ] GitHub Actions CI is green.
- [ ] Pages workflow is green and the public URL returns HTTP 200.
- [ ] If deploying Streamlit, add the live URL to README and `docs/index.html`.
- [ ] Record the model version, threshold, and date with every exported campaign list.

## 8. Security and privacy

- The app requires no credentials and stores no customer data.
- Uploaded batch files are processed in memory by the Streamlit session; they are not written to disk by this project.
- Do not upload real customer data to a public demo app. Use anonymised extracts.
- Review the model card before using scores in a customer-facing process.
- The repository contains only the public, anonymised Telco dataset.
