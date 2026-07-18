# Clean Architecture Design Document

## Overview

This document explains the Clean Architecture design used in the Diet Health Predictor project and provides justification for the architectural choices.

## What is Clean Architecture?

Clean Architecture is a software architecture approach that emphasizes:

1. **Independence of frameworks** - Framework choice should not limit business logic
2. **Testability** - Core business logic can be tested without external dependencies
3. **Independence of UI** - Business logic is independent of presentation layer
4. **Independence of database** - Business logic doesn't know or care about data source
5. **Independence of any external agency** - Business logic is isolated from the outside world

## Project Layers

### 1. Domain Layer (`domain/`)

**Responsibility:** Contain pure business logic and entities

**Contents:**
```python
# Core entities
- Person
- DietAssessment
- NutritionData

# Enumerations
- HealthStatus
- DietType
- ActivityLevel
```

**Key Characteristics:**
- ✅ NO external dependencies (no imports from other layers)
- ✅ NO framework-specific code
- ✅ Pure Python dataclasses and enums
- ✅ Business rule methods (e.g., `is_healthy()`, `needs_intervention()`)
- ✅ Value objects and entities

**Why this approach:**
- Domain code can be tested in isolation
- Easy to change frameworks without touching business logic
- Highly reusable across different projects

**Example:**
```python
@dataclass
class Person:
    person_id: str
    age: int
    bmi: float
    
    def is_healthy(self) -> bool:
        """Pure business logic: BMI-based health determination"""
        return 18.5 <= self.bmi < 25.0
```

### 2. Application Layer (`application/`)

**Responsibility:** Orchestrate use cases and business logic

**Contents:**
```python
# Use Cases
- LoadHealthDietDataUseCase
- AnalyzeHealthStatsUseCase
```

**Key Characteristics:**
- ✅ Depends on Domain layer only
- ✅ Orchestrates domain objects and business rules
- ✅ Converts raw data into domain objects
- ✅ Implements application-specific business rules
- ❌ Does NOT interact directly with external systems

**Why this approach:**
- Clear separation between business rules and infrastructure
- Easy to test: mock infrastructure, test use cases
- Use cases become reusable across different presentations

**Example:**
```python
class LoadHealthDietDataUseCase:
    """Use case: Load data and transform to domain objects"""
    
    def execute(self) -> list[DietAssessment]:
        # Get raw data from infrastructure
        df = self.data_loader.load()
        
        # Transform to domain objects
        assessments = [
            self._row_to_assessment(row) 
            for row in df.iterrows()
        ]
        
        return assessments
```

### 3. Infrastructure Layer (`infrastructure/`)

**Responsibility:** Handle external I/O operations

**Contents:**
```python
# Data loaders
- DataLoader
- HealthDietDataLoader

# External services
- (Database adapters - to be added)
- (API clients - to be added)
```

**Key Characteristics:**
- ✅ Implements details of external systems
- ✅ Adapts external data formats to domain models
- ✅ Handles file I/O, database queries, API calls
- ✅ Can be easily swapped/mocked for testing

**Why this approach:**
- All "messy" external details are isolated here
- Easy to change data source (file → database → API)
- Application layer never needs to know implementation details

**Example:**
```python
class DataLoader:
    """Infrastructure: Reads from CSV files"""
    
    def load(self) -> pd.DataFrame:
        # Read CSV file
        return pd.read_csv(self.data_path)
```

### 4. Presentation Layer (`presentation/`)

**Responsibility:** Expose functionality to external consumers

**Contents:**
```python
# API interface
- HealthDietAPI

# Entry points
- main()

# (Controllers - to be added for FastAPI)
```

**Key Characteristics:**
- ✅ Depends on Application layer
- ✅ Formats responses for consumers
- ✅ Handles HTTP requests (when API added)
- ✅ Simple, delegating to use cases

**Why this approach:**
- Clean separation between API and business logic
- Easy to add multiple presentation layers (CLI, API, Web UI)
- Business logic remains independent of how it's exposed

**Example:**
```python
class HealthDietAPI:
    """Presentation: User-facing API"""
    
    def load_data(self):
        # Delegate to use case
        use_case = LoadHealthDietDataUseCase(self.loader)
        return use_case.execute()
```

### 5. Configuration Layer (`config/`)

**Responsibility:** Manage environment-specific settings

**Contents:**
```python
# Settings models
- DataConfig
- ModelConfig
- APIConfig
- StorageConfig
- Settings

# Configuration loading
- from_yaml()
- get_settings()
```

**Key Characteristics:**
- ✅ Multi-environment support (dev, staging, prod)
- ✅ YAML-based configuration
- ✅ Environment variable overrides
- ✅ Validation with Pydantic

**Why this approach:**
- Configuration is centralized and validated
- Easy to manage different settings per environment
- No hardcoded values in code

**Example:**
```yaml
# config/settings.dev.yaml
environment: development
debug: true
data:
  raw_data_path: "data/healthy_diet_calorie_intake.csv"
  sample_size: null
```

## Data Flow Example

### Scenario: "Load and analyze health data"

```
1. Presentation Layer (HealthDietAPI)
   └─> user calls: api.load_data()

2. Application Layer (LoadHealthDietDataUseCase)
   └─> orchestrates: raw_data → domain_objects

3. Infrastructure Layer (HealthDietDataLoader)
   └─> reads: CSV file → raw DataFrame

4. Domain Layer (DietAssessment, Person, etc.)
   └─> represents: business entities

5. Back to Application Layer
   └─> transforms: DataFrame rows → domain objects

6. Back to Presentation Layer
   └─> returns: List[DietAssessment] to caller
```

### Code Flow:
```python
# Step 1: User calls API
api = HealthDietAPI()
assessments = api.load_data()  # ← Presentation

# Step 2: API delegates to use case (Application)
use_case = LoadHealthDietDataUseCase(loader)
return use_case.execute()

# Step 3: Use case gets data from infrastructure
df = self.data_loader.load()  # ← Infrastructure (CSV read)

# Step 4: Transform to domain objects (Application)
for row in df.iterrows():
    assessment = self._row_to_assessment(row)  # Domain

# Step 5: Return domain objects
return [assessment1, assessment2, ...]  # ← Domain
```

## Environment-Based Configuration Strategy

### Why Multi-Environment?

Different environments have different requirements:

| Aspect | Development | Staging | Production |
|--------|-------------|---------|-----------|
| **Logging** | DEBUG (verbose) | INFO | WARNING (minimal) |
| **Data** | Full dataset / sample | Full dataset | Production data |
| **Paths** | Local relative paths | Mounted volumes | Absolute paths |
| **Cache** | Disabled | Enabled | Enabled |
| **Debug** | true | false | false |

### Implementation

```
config/
├── settings.dev.yaml      # Development settings
├── settings.staging.yaml  # Staging settings
└── settings.prod.yaml     # Production settings
```

**Loading logic:**
```python
# Reads ENVIRONMENT env var, then loads appropriate YAML
settings = Settings.from_yaml()  # Loads based on ENVIRONMENT
```

**Advantages:**
1. ✅ Configuration is separate from code
2. ✅ Easy to manage different settings per environment
3. ✅ No need to modify code between environments
4. ✅ Secrets can be overridden via environment variables

## Dependency Injection Pattern

### Current Approach: Constructor Injection

```python
class LoadHealthDietDataUseCase:
    def __init__(self, data_loader: HealthDietDataLoader):
        self.data_loader = data_loader  # Injected dependency
```

**Benefits:**
- ✅ Easy to test (inject mock loader)
- ✅ Dependencies are explicit
- ✅ No hidden dependencies
- ✅ Can work with different implementations

**Example Test:**
```python
def test_load_data():
    mock_loader = MockDataLoader()
    use_case = LoadHealthDietDataUseCase(mock_loader)
    result = use_case.execute()
    assert len(result) > 0
```

## Phase 2: Data Preprocessing & Feature Engineering

Phase 2 grows the Infrastructure and Application layers to turn raw data into
model-ready features, without touching the Domain or Presentation layers' existing
contracts.

### Infrastructure Layer additions (`infrastructure/preprocessing.py`)

- **`DataCleaner`** - `drop_duplicates()` (stateless, run on the full dataset
  before splitting so a duplicate row can't land in both train and test) plus
  `fit()`/`transform()` for imputation (median for numeric columns, mode for
  categorical) and outlier clipping (IQR method). Like `FeatureTransformer`,
  the imputation values and outlier bounds are learned from the training
  split only, then applied to test -- fitting them on the full dataset
  before splitting would leak test-set statistics into training data (a real
  bug in an earlier version of this pipeline: outlier bounds computed on the
  full dataset actually changed 149 cells on the real dataset)
- **`FeatureEngineer`** - derives 14 new columns:
  - Calorie balance: `Calorie_Balance`, `Calorie_Deviation_Pct`
  - Macro composition: `Protein_Ratio`, `Carb_Ratio`, `Fat_Ratio`, `Total_Macros_g`
  - Hydration: `Adequate_Water_Intake`, `Water_Intake_ml_per_kg`
  - Age bucketing: `Age_Group`
  - Metabolism: `BMR` (Mifflin-St Jeor), `Activity_Calorie_Multiplier` (stated
    calorie requirement / BMR — cross-checks the declared `Activity_Level`
    against actual metabolic demand)
  - Body composition: `Ideal_Weight_kg` / `Weight_Deviation_kg` (Devine formula,
    an alternative signal to BMI), `Protein_per_kg_Bodyweight`

  These vectorize (over a DataFrame) the same formulas expressed as single-record
  business rules on `Person` / `NutritionData` / `DietAssessment` in the Domain
  layer — batch processing needs the pandas version, but the rule itself lives
  conceptually in the Domain layer
- **`FeatureTransformer`** - wraps a scikit-learn `ColumnTransformer`
  (`StandardScaler` + `OneHotEncoder`); fit on train, reused on test to avoid
  leakage; can be saved/loaded via `joblib` for reuse at inference time
- **`DataSplitter`** - stratified `train_test_split` wrapper
- **`ProcessedDataWriter`** - persists `X_train/X_test/y_train/y_test.csv` and the
  fitted transformer to the configured `processed_data_path`

### Application Layer additions (`application/feature_engineering.py`)

- **`PreprocessDataUseCase`** - orchestrates the pipeline: load → dedup → split
  → clean (fit on train only) → engineer → encode/scale (fit on train only) →
  persist. Deduplication runs before the split; everything that learns
  statistics from data runs after it, fit on train only
- **`LEAKING_NUMERIC_COLUMNS`** - `Height_cm`, `Weight_kg`, `BMI`, and every
  engineered column derived from them (`BMR`, `Activity_Calorie_Multiplier`,
  `Protein_per_kg_Bodyweight`, `Water_Intake_ml_per_kg`, `Ideal_Weight_kg`,
  `Weight_Deviation_kg`), excluded from `NUMERIC_FEATURES`. `Health_Status`
  turns out to be an *exact* deterministic function of `BMI` in this dataset
  (WHO BMI thresholds, zero exceptions across all 6000 rows) — training on
  `BMI` gives a trivial 100% accuracy, and `Height_cm`/`Weight_kg` alone
  reconstruct it almost perfectly (~98%). These columns are still computed by
  `FeatureEngineer` (useful for analysis) but never reach the model. Full
  investigation in `notebooks/03_model_training.ipynb`, section 1
- **`PreprocessingResult`** - dataclass carrying `X_train`, `X_test`, `y_train`,
  `y_test`, `feature_names`, and `transformer_path` back to the caller

### Presentation Layer addition

- **`HealthDietAPI.preprocess_data()`** - thin delegation to `PreprocessDataUseCase`,
  same pattern as `load_data()` / `get_health_statistics()`

## Phase 3: Model Training, Cross-Validation & Comparison

Only boosting classifiers are supported (XGBoost, CatBoost) -- deliberately,
not "any scikit-learn-compatible model": both track a per-iteration
train/eval metric, which the rest of this phase is built around.

### Infrastructure Layer additions (`infrastructure/models/`)

- **`BaseModelWrapper`** - abstract base defining the shared contract: `fit`
  (optionally given a validation set), `predict`, `predict_proba`, `save`,
  `load`, `get_evals_result`. Subclasses implement `_build_model()` and `_fit()`
- **`XGBoostWrapper`** / **`CatBoostWrapper`** - one module each. Neither
  hardcodes a single hyperparameter default beyond `random_state`: every
  other value (`n_estimators`/`iterations`, `max_depth`/`depth`,
  `learning_rate`, `eval_metric`, `verbose`, ...) comes from
  `settings.model.{xgboost,catboost}_params` (YAML) via `**hyperparameters`;
  an unset key falls back to that library's own built-in default, never a
  value invented in this codebase
- **`get_evals_result()`** - normalizes each library's own eval-curve API
  (`XGBClassifier.evals_result()`'s `validation_0`/`validation_1`,
  `CatBoostClassifier.get_evals_result()`'s `learn`/`validation`) into a
  common `{"train": {...}, "validation": {...}}` shape
- CatBoost's `.predict()` returns shape `(n, 1)`; `CatBoostWrapper` flattens
  it to 1D to match `XGBoostWrapper`

### Application Layer additions

- **`model_training.py`**
  - `ModelType` / `MODEL_WRAPPER_REGISTRY` - the 2 supported models and their wrapper classes
  - `compute_classification_metrics()` - accuracy, precision/recall/f1 (macro),
    MCC (Matthews Correlation Coefficient), and a confusion matrix pinned to
    every class the `LabelEncoder` knows about (so it's always the same shape
    as `class_labels`, even when a class is absent from a particular small
    test/fold batch)
  - `TrainModelUseCase` - label-encodes the target once, trains one model type
    against the held-out test set as its eval_set (for train/eval curves),
    evaluates, and persists the model + label encoder. `hyperparameters` can
    be passed to `HealthDietAPI.train_model()` explicitly to override the
    YAML-configured ones for a single call
- **`model_evaluation.py`**
  - `CrossValidateModelUseCase` - stratified k-fold CV via an explicit fold
    loop (our wrappers aren't scikit-learn estimators, so
    `cross_val_score`/`clone()` can't drive them); reports per-fold metrics
    plus their mean/std. Nothing is persisted -- it's a stability check, not
    a training run
  - `CompareModelsUseCase` - runs `TrainModelUseCase` for every registered
    `ModelType` on the same train/test split, each with its own
    hyperparameters
  - `best_model()` - picks the `ModelType` with the highest value of a chosen
    metric from a `CompareModelsUseCase` result. Defaults to **`mcc`**, not
    `f1_macro` or `accuracy` -- MCC stays meaningful under class imbalance,
    making it the more trustworthy tie-breaker on this dataset
  - `save_best_model()` - calls `best_model()`, then copies the winner's
    `model.joblib` / `label_encoder.joblib` into `output_dir/best/` alongside
    a `metadata.json` (`model_type`, `selection_metric`,
    `selection_metric_value`) -- a canonical location a downstream consumer
    (e.g. a future Phase 4 API) can load "the" model from without knowing
    which `ModelType` won. Both models are still saved under their own
    `output_dir/<model_type>/` too, by `TrainModelUseCase`

### Presentation Layer additions

- **`HealthDietAPI.train_model(model_type, preprocessing_result, hyperparameters=None)`**
  - `hyperparameters`, when given, replaces `settings.model.{model_type}_params`
    entirely for that call (e.g. for a quick notebook experiment)
- **`HealthDietAPI.cross_validate_model(model_type, preprocessing_result, n_splits=5)`**
  - recombines `preprocessing_result.X_train`/`X_test` (and `y_train`/`y_test`)
    before cross-validating, since CV should see every available row, not
    just the train split
- **`HealthDietAPI.compare_models(preprocessing_result)`**
- **`HealthDietAPI.select_best_model(comparison, metric=None)`**
  - resolves `metric` from `settings.model.selection_metric` ("mcc") when not
    given, and delegates to `save_best_model()`
- All hyperparameter-driven methods read from `settings.model.{xgboost,catboost}_params`
  via a shared `_hyperparameters_for(model_type)` helper

## Future Improvements

### Phase 4: API Layer
- [ ] FastAPI endpoints
- [ ] Request/response models
- [ ] Error handling and status codes
- [ ] API documentation (OpenAPI)

### Phase 5: Testing
- [ ] Unit tests for each layer
- [ ] Integration tests
- [ ] E2E tests
- [ ] Mocking strategies

### Phase 6: Scalability
- [ ] Database integration
- [ ] Caching layer
- [ ] Async operations
- [ ] Distributed processing

## Design Principles Applied

### SOLID Principles

1. **Single Responsibility Principle (SRP)**
   - Each class has one reason to change
   - E.g., DataLoader only handles reading, not processing

2. **Open/Closed Principle (OCP)**
   - Open for extension, closed for modification
   - New use cases don't modify existing code

3. **Liskov Substitution Principle (LSP)**
   - Subclasses can substitute parent classes
   - Different loaders can replace HealthDietDataLoader

4. **Interface Segregation Principle (ISP)**
   - Clients depend on specific interfaces
   - Not forced to depend on irrelevant methods

5. **Dependency Inversion Principle (DIP)**
   - Depend on abstractions, not concrete implementations
   - Use cases depend on interfaces, not specific loaders

### DRY (Don't Repeat Yourself)
- Configuration is centralized in YAML files
- Settings validation is centralized in Pydantic models

### Separation of Concerns
- Each layer has a specific responsibility
- Layers communicate through well-defined interfaces

## Testing Strategy

```python
# Test domain layer (no external dependencies)
def test_person_is_healthy():
    person = Person(..., bmi=22.0)
    assert person.is_healthy() == True

# Test application layer (mock infrastructure)
def test_load_data_use_case():
    mock_loader = Mock()
    mock_loader.load.return_value = pd.DataFrame(...)
    use_case = LoadHealthDietDataUseCase(mock_loader)
    result = use_case.execute()
    assert len(result) > 0

# Test infrastructure layer (real I/O)
def test_data_loader():
    loader = HealthDietDataLoader("data/test.csv")
    df = loader.load()
    assert len(df) > 0

# Test presentation layer (mock application)
def test_api():
    api = HealthDietAPI()
    stats = api.get_health_statistics([...])
    assert 'total_assessments' in stats
```

## Configuration as Code (IaC)

All infrastructure and environment configurations are defined in code and YAML:

- ✅ Version controlled
- ✅ Reproducible across environments
- ✅ Easy to audit changes
- ✅ Can be automated with CI/CD

## Deployment Considerations

### Multi-Environment Deployment

```bash
# Development
export ENVIRONMENT=development
python -m diet_health_predictor.presentation

# Staging
export ENVIRONMENT=staging
python -m diet_health_predictor.presentation

# Production
export ENVIRONMENT=production
python -m diet_health_predictor.presentation
```

### Docker Deployment (Future)

```dockerfile
ENV ENVIRONMENT=production
CMD python -m diet_health_predictor.presentation
```

---

**Last Updated:** July 2026
**Version:** 0.1.0
