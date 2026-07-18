# Notebooks Directory 📓

This directory contains Jupyter notebooks for exploring and testing the Diet-Health-Predictor architecture.

## Available Notebooks

### 📘 `01_architecture_demo.ipynb`

**Purpose:** Comprehensive demonstration of all five layers of the Clean Architecture

**What you'll learn:**
- How to load and use settings from different environments (dev/staging/prod)
- Infrastructure layer: Data loading from CSV with validation
- Domain layer: Creating and using pure business entities
- Application layer: Orchestrating use cases
- Presentation layer: Using the high-level API
- Comparing configurations across environments
- Detailed data exploration and statistics

**Key Sections:**
1. **Setup & Configuration** - Initialize paths and environment
2. **Configuration Layer** - Load YAML settings with Pydantic
3. **Infrastructure Layer** - Load data using specialized loaders
4. **Domain Layer** - Create and use business entities
5. **Application Layer** - Execute use cases
6. **Presentation Layer** - Use the high-level API
7. **Multi-Environment Configuration** - Compare dev/staging/prod settings
8. **Data Exploration** - Analyze the dataset

**Execution Time:** ~2-3 minutes

---

### 📗 `02_data_preprocessing.ipynb`

**Purpose:** Phase 2 walkthrough — cleaning, feature engineering, splitting, drift-checking, and encoding/scaling

**What you'll learn:**
- Infrastructure layer: `DataCleaner`, `FeatureEngineer`, `FeatureTransformer`, `DataSplitter`, `DriftDetector`, `ProcessedDataWriter`
- Application layer: `PreprocessDataUseCase` orchestrating the full pipeline
- Presentation layer: running everything via `HealthDietAPI.preprocess_data()`
- How the derived features (`Age_Group`, `Calorie_Balance`, macro ratios, etc.) relate to the target
- Why `BMI`/`Height_cm`/`Weight_kg` and everything derived from them are computed but
  excluded from the model's feature set (`LEAKING_NUMERIC_COLUMNS`) -- full investigation
  in `03_model_training.ipynb`, section 1
- Checking the train/test split itself for drift (PSI + KS test) right after it's created,
  on the raw/engineered columns (numeric *and* categorical in one pass)

**Key Sections:**
1. **Setup & Configuration**
2. **Load Raw Data** - infrastructure layer
3. **Data Cleaning** - `DataCleaner`
4. **Feature Engineering** - `FeatureEngineer`
5. **Train/Test Split** - `DataSplitter`
6. **Check for Data Drift** - `DriftDetector`, train vs. test on the raw/engineered columns
7. **Encoding & Scaling** - `FeatureTransformer`
8. **Full Pipeline** - `PreprocessDataUseCase` via `HealthDietAPI`
9. **Quick Feature Sanity Checks**

**Execution Time:** ~1-2 minutes

---

### 📙 `03_model_training.ipynb`

**Purpose:** Phase 3 walkthrough — investigating data leakage, then training, evaluating, and comparing boosting models on `Health_Status`

**What you'll learn:**
- How to detect target leakage: checking whether `Health_Status` is a deterministic
  function of `BMI` (it is), and measuring how far the leak spreads via `Height_cm`/`Weight_kg`
  and their derived features, with empirical before/after accuracy comparisons
- How to check for data drift between train and test (PSI + KS test) before
  trusting any test-set metric
- Infrastructure layer: `XGBoostWrapper`, `CatBoostWrapper` (uniform fit/predict/save/load/get_evals_result contract), `DriftDetector`
- Application layer: `TrainModelUseCase`, `CrossValidateModelUseCase`, `CompareModelsUseCase`, `best_model()`, `save_best_model()`, `AnalyzeDataDriftUseCase`
- Presentation layer: `HealthDietAPI.train_model()` / `cross_validate_model()` / `compare_models()` / `select_best_model()` / `analyze_data_drift()`
- Reading train/eval curves to diagnose overfitting
- Why hyperparameters live entirely in `settings.model.{xgboost,catboost}_params` (YAML), never hardcoded in the wrapper code

**Key Sections:**
- **Setup & Configuration**
1. **Data Leakage Investigation** - BMI-to-Health_Status determinism check, empirical
   accuracy comparison across feature subsets, the `LEAKING_NUMERIC_COLUMNS` fix
2. **Load & Preprocess Data** - Phase 2 recap via `HealthDietAPI.preprocess_data()`, now leak-free
3. **Data Drift Check** - `HealthDietAPI.analyze_data_drift()` (PSI + KS test, X_train vs. X_test)
4. **Train a Single Model** - `HealthDietAPI.train_model()`
5. **Train/Eval Curves** - per-iteration train vs. validation metric
6. **Train CatBoost Too** - same pipeline, second model
7. **Cross-Validation** - `HealthDietAPI.cross_validate_model()`
8. **Compare & Select Best Model** - `HealthDietAPI.compare_models()` + `select_best_model()` (by MCC, by default)
9. **Confusion Matrix**

**Execution Time:** ~1-2 minutes

---

### 📕 `04_api_testing.ipynb`

**Purpose:** Phase 4 walkthrough — serving the persisted best model over HTTP with FastAPI

**What you'll learn:**
- The inference-time gap Phase 4 had to close first: `DataCleaner.transform()`
  now skips columns absent from a single incoming record (e.g. `Person_ID`,
  `Health_Status`), and `DataCleaner` itself is now persisted (`save()`/`load()`)
  alongside `FeatureTransformer` so prediction-time cleaning reuses the exact
  training-derived impute values/outlier bounds
- Application layer: `PredictHealthStatusUseCase` — replays
  `clean -> engineer -> encode/scale` on one raw record before calling
  `model.predict_proba()`
- Presentation layer: `HealthDietAPI.predict_health_status()` /
  `get_best_model_info()`, then the FastAPI app itself
  (`presentation/api_app.py`, `presentation/schemas.py`)
- Testing the API in-process with `TestClient` — no server to start
- A sanity check on the Phase 3 leakage fix: changing only `Weight_kg`/`BMI`
  barely moves the prediction (both are excluded model features), while a
  genuine calorie surplus does

**Key Sections:**
- **Setup & Configuration**
1. **Train & Persist a Best Model** - recap of Phases 2-3 (`preprocess_data()`,
   `compare_models()`, `select_best_model()`)
2. **The Prediction Pipeline Directly** - `HealthDietAPI.predict_health_status()`
3. **The FastAPI App** - routes, schemas, OpenAPI docs
4. **Testing the API** - `TestClient`, valid and invalid (422) requests
5. **Running the Server for Real** - `uvicorn`, `curl`

**Execution Time:** ~1-2 minutes

---

## How to Run Notebooks

### Prerequisites
```bash
# Install dependencies (if not done yet)
poetry install

# Activate environment
poetry shell
```

### Option 1: VS Code
1. Open the notebook file in VS Code
2. Select Python interpreter (should be poetry venv)
3. Run cells individually or all at once

### Option 2: Jupyter Lab
```bash
# Install Jupyter (if not in pyproject.toml dev dependencies)
poetry add --group dev jupyter

# Start Jupyter Lab
jupyter lab
```

### Option 3: Command Line
```bash
# Run specific notebook
jupyter nbconvert --to notebook --execute 01_architecture_demo.ipynb
```

---

## Notebook Structure

Each notebook follows this structure:

```
📓 Notebook
├── 📝 Title & Overview
├── 🔧 Setup & Configuration
├── 📚 Theory (markdown explanation)
├── 💻 Implementation (code cells)
├── ✅ Verification (output inspection)
└── 🚀 Next Steps
```

---

## Tips for Using These Notebooks

### 1. **Run Cells Sequentially**
- Notebooks depend on earlier cells being executed
- If you skip a cell, later cells may fail
- Use "Run All" or run cells in order

### 2. **Modify and Experiment**
- These notebooks are meant to be interactive
- Try changing parameters and see what happens
- Add new cells to test your hypotheses

### 3. **Use for Learning**
- Read the markdown explanations first
- Understand what each section does
- Experiment with the code

### 4. **Environment Variables**
- Edit the `ENVIRONMENT` variable to test different configurations
- Changes will be reflected in the API behavior

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'diet_health_predictor'"
**Solution:** Make sure you're:
1. In the poetry environment (`poetry shell`)
2. Running the notebook from the notebooks folder
3. The setup cell with `sys.path.insert()` was executed

### "FileNotFoundError: data/healthy_diet_calorie_intake.csv"
**Solution:** 
1. Verify the data file exists at the project root
2. Check the path in the notebook matches your setup
3. Run from the correct working directory

### "YAML not found: settings.dev.yaml"
**Solution:**
1. Ensure config files exist in `config/` directory
2. Check file names match exactly: `settings.{env}.yaml`
3. Verify the project structure is intact

---

## Future Notebooks

Planned notebooks for upcoming phases:

- `05_deployment_guide.ipynb` - Deploying to different environments

---

## Quick Reference

| Layer | File | Class |
|-------|------|-------|
| **Configuration** | `config/__init__.py` | `Settings`, `get_settings()` |
| **Infrastructure** | `infrastructure/data_loader.py` | `HealthDietDataLoader` |
| **Infrastructure** | `infrastructure/preprocessing.py` | `DataCleaner`, `FeatureEngineer`, `FeatureTransformer`, `DataSplitter`, `ProcessedDataWriter` |
| **Domain** | `domain/__init__.py` | `Person`, `DietAssessment` |
| **Application** | `application/use_cases.py` | `LoadHealthDietDataUseCase` |
| **Application** | `application/feature_engineering.py` | `PreprocessDataUseCase` |
| **Infrastructure** | `infrastructure/models/` | `XGBoostWrapper`, `CatBoostWrapper` |
| **Application** | `application/model_training.py` | `TrainModelUseCase` |
| **Application** | `application/model_evaluation.py` | `CrossValidateModelUseCase`, `CompareModelsUseCase`, `best_model()`, `save_best_model()` |
| **Infrastructure** | `infrastructure/drift.py` | `DriftDetector` |
| **Application** | `application/data_drift.py` | `AnalyzeDataDriftUseCase` |
| **Application** | `application/prediction.py` | `PredictHealthStatusUseCase` |
| **Presentation** | `presentation/__init__.py` | `HealthDietAPI` |
| **Presentation** | `presentation/schemas.py` | `PredictionRequest`, `PredictionResponse` |
| **Presentation** | `presentation/api_app.py` | FastAPI `app` (`/health`, `/model/info`, `/predict`) |

---

## Notes

- **Python Version:** 3.14+
- **Main Dependencies:** pandas, pydantic, pyyaml, scikit-learn, joblib, xgboost, catboost, scipy
- **Kernel:** Python 3.x (poetry environment)
- **Working Directory:** Should be project root or notebooks folder

---

## Support

For issues or questions:
1. Check the main `README.md`
2. Review `ARCHITECTURE.md` for design details
3. Inspect the source code in `src/`
4. Add debug cells as needed

Happy exploring! 🚀
