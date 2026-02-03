# Preprocessing Pipeline Refactoring: Data Leakage Fix and sklearn Best Practices

## Executive Summary

This document details a critical refactoring of the preprocessing pipeline in [04-preprocessing.ipynb](04-preprocessing.ipynb) that addresses:

1. **Data leakage bug** in temporal feature engineering
2. **Inference reproducibility** issues with the saved pipeline
3. **sklearn best practices** compliance for production-ready ML code

The refactoring encapsulates all transformations in a complete sklearn pipeline using a custom transformer, fixing subtle data leakage and enabling true end-to-end inference.

---

## Problem Analysis

### Problem 1: Data Leakage in `days_since_start` Feature

#### Previous Implementation (INCORRECT)

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations."""
    df = df.copy()
    
    # PROBLEM: Computes min_date separately for each split
    min_date = df["date_parsed"].min()  # ← Different value per split!
    df["days_since_start"] = (df["date_parsed"] - min_date).dt.days
    
    # ... other features ...
    return df

# Applied separately to each split
train_eng = engineer_features(train_df)  # min_date from train
val_eng = engineer_features(val_df)      # min_date from val (LEAK!)
test_eng = engineer_features(test_df)    # min_date from test (LEAK!)
```

#### Why This Is Data Leakage

The `days_since_start` feature is intended to capture temporal trends (market appreciation/depreciation over time). However, when computed independently for each split:

| Split | `min_date` | `days_since_start` for same date |
|-------|-----------|----------------------------------|
| Train | 2014-05-02 | 365 |
| Val   | 2015-02-16 | **0** ← Wrong! Should be ~365 |
| Test  | 2015-04-01 | **0** ← Wrong! Should be ~410 |

**Consequence**: The model learns that `days_since_start=0` means "start of dataset" during training, but at inference time on validation/test data, `days_since_start=0` actually means "middle/end of dataset". This creates:

1. **Inconsistent feature semantics** across splits
2. **Information leakage** - validation/test dates are re-centered, effectively "informing" the model about future time periods
3. **Invalid predictions** at inference on new data

#### Concrete Example

Suppose we have sales from 2014-05-02 to 2015-05-27:

**Before (INCORRECT)**:
- Training set: 2014-05-02 to 2015-02-15
  - `min_date = 2014-05-02`
  - House sold on 2015-02-15: `days_since_start = 289`
  
- Validation set: 2015-02-16 to 2015-04-01
  - `min_date = 2015-02-16` ← **WRONG REFERENCE**
  - House sold on 2015-03-01: `days_since_start = 13` ← Should be ~303!
  
- Test set: 2015-04-02 to 2015-05-27
  - `min_date = 2015-04-02` ← **WRONG REFERENCE**
  - House sold on 2015-05-01: `days_since_start = 29` ← Should be ~365!

The model trained on the training set expects `days_since_start` around 289 for late-dataset sales, but validation/test sets present small values instead, breaking the learned pattern.

---

### Problem 2: Incomplete Saved Pipeline

#### Previous Implementation

```python
# Feature engineering happens OUTSIDE the pipeline
train_eng = engineer_features(train_df)
val_eng = engineer_features(val_df)
test_eng = engineer_features(test_df)

# Separate X/y
X_train = train_eng.drop(columns=["price"])
y_train = train_eng["price"]
# ... (similar for val/test)

# Pipeline only includes log+scale, NOT feature engineering
preprocessor = ColumnTransformer([
    ("log", log_pipeline, log_features),
    ("scale", StandardScaler(), scale_features),
    ("passthrough", "passthrough", passthrough_features)
])

preprocessor.fit(X_train)  # Only fits scaling, not feature engineering

# Save INCOMPLETE pipeline
joblib.dump(preprocessor, "preprocessor.joblib")
```

#### Why This Is Problematic

The saved `preprocessor.joblib` **cannot process raw data** because:

1. **Missing feature engineering**: The transformer expects features like `days_since_start`, `house_age`, `was_renovated` to already exist
2. **Missing reference values**: Even if you manually run `engineer_features()` at inference, you don't have the training `min_date` value
3. **Manual intervention required**: Every inference requires manually running feature engineering with hardcoded reference values

**Inference failure scenario**:
```python
# Load saved pipeline
preprocessor = joblib.load("preprocessor.joblib")

# Try to use on new data
new_df = pd.read_csv("new_houses.csv")
new_df["date_parsed"] = pd.to_datetime(new_df["date"].str[:8])

# This FAILS - features missing!
preprocessor.transform(new_df)
# KeyError: "['days_since_start', 'house_age', ...] not found in columns"
```

---

### Problem 3: Not sklearn Best Practices

The previous approach violated sklearn conventions:

1. **Transformations split across pipeline and external code** - Feature engineering outside, scaling inside
2. **No `.fit()` for feature engineering** - Reference values (like `min_date`) not learned from training data
3. **Manual state management** - User must track `min_date` separately from the pipeline
4. **Not composable** - Cannot integrate into larger `Pipeline` or `GridSearchCV`

This makes the code:
- **Harder to maintain** - Two separate transformation systems
- **Prone to errors** - Easy to forget feature engineering step
- **Not production-ready** - Doesn't follow ML engineering standards

---

## Solution: Custom sklearn Transformer

### Implementation

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for house price prediction.
    
    Fits reference values (min_date) on training data to avoid data leakage.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer by learning reference values from training data.
        
        Parameters
        ----------
        X : DataFrame
            Training data with 'date_parsed' column.
        y : array-like, optional
            Target variable (not used).
            
        Returns
        -------
        self
        """
        # Store the minimum date from TRAINING data only
        # This becomes a fitted parameter, like mean in StandardScaler
        self.min_date_ = X["date_parsed"].min()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering using fitted reference values.
        
        Uses self.min_date_ learned from training, ensuring consistency.
        """
        X = X.copy()
        
        # Use FITTED min_date_ (from training) for all splits
        X["days_since_start"] = (X["date_parsed"] - self.min_date_).dt.days
        
        # Row-level features (no fitting needed)
        sale_year = X["date_parsed"].dt.year
        X["house_age"] = sale_year - X["yr_built"]
        X["was_renovated"] = (X["yr_renovated"] > 0).astype(int)
        X["years_since_renovation"] = np.where(
            X["yr_renovated"] > 0,
            sale_year - X["yr_renovated"],
            0
        )
        X["basement_ratio"] = X["sqft_basement"] / X["sqft_living"].replace(0, 1)
        X["living_vs_neighbors"] = X["sqft_living"] / X["sqft_living15"].replace(0, 1)
        X["lot_vs_neighbors"] = X["sqft_lot"] / X["sqft_lot15"].replace(0, 1)
        
        # Drop columns
        columns_to_drop = [
            "id", "date", "date_parsed", "zipcode", 
            "yr_built", "yr_renovated"
        ]
        X = X.drop(columns=columns_to_drop)
        
        return X
```

### Key Design Decisions

#### 1. Fitted Parameter: `min_date_`

Following sklearn convention, fitted parameters end with underscore (`_`):

```python
def fit(self, X, y=None):
    self.min_date_ = X["date_parsed"].min()  # ← Fitted parameter
    return self
```

This is analogous to:
- `StandardScaler.mean_` (fitted on training)
- `LabelEncoder.classes_` (fitted on training)
- `PCA.components_` (fitted on training)

**Benefits**:
- Clear distinction between hyperparameters (set at initialization) and fitted values (learned from data)
- Accessible after fitting: `transformer.min_date_`
- Preserved during serialization with `joblib`

#### 2. Separation of Fit and Transform Logic

```python
def fit(self, X, y=None):
    """Learn reference values from TRAINING data only"""
    self.min_date_ = X["date_parsed"].min()
    return self

def transform(self, X):
    """Apply transformations using fitted references"""
    X["days_since_start"] = (X["date_parsed"] - self.min_date_).dt.days
    # ... other transformations
    return X
```

This enables the correct sklearn workflow:

```python
# Fit on training data
transformer.fit(X_train)         # Learns min_date_ = 2014-05-02

# Transform all splits using SAME reference
X_train_t = transformer.transform(X_train)  # Uses 2014-05-02
X_val_t = transformer.transform(X_val)      # Uses 2014-05-02 ✓
X_test_t = transformer.transform(X_test)    # Uses 2014-05-02 ✓
```

#### 3. Complete Pipeline Integration

```python
# Numeric preprocessing (log + scale)
numeric_preprocessor = ColumnTransformer([
    ("log", log_pipeline, log_features),
    ("scale", StandardScaler(), scale_features),
    ("passthrough", "passthrough", passthrough_features)
])

# Complete pipeline: feature engineering → numeric preprocessing
full_pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", numeric_preprocessor)
])
```

Now we have a **single, complete pipeline** that:
1. Takes raw data with `date_parsed` column
2. Engineers features using fitted reference values
3. Applies log transformation and scaling
4. Returns final processed features

---

## Detailed Comparison: Before vs. After

### Workflow Comparison

#### Before (INCORRECT)

```python
# 1. Manual feature engineering (separate for each split)
train_eng = engineer_features(train_df)  # min_date from train
val_eng = engineer_features(val_df)      # min_date from val ← LEAK!
test_eng = engineer_features(test_df)    # min_date from test ← LEAK!

# 2. Separate X/y
X_train = train_eng.drop(columns=["price"])
y_train = train_eng["price"]
# ... (similar for val/test)

# 3. Partial pipeline (only scaling)
preprocessor = ColumnTransformer([...])
preprocessor.fit(X_train)

# 4. Transform
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# 5. Save INCOMPLETE pipeline
joblib.dump(preprocessor, "preprocessor.joblib")
```

**Problems**:
- ❌ Data leakage in `days_since_start`
- ❌ Incomplete saved pipeline
- ❌ Manual state management
- ❌ Not composable

#### After (CORRECT)

```python
# 1. Separate X/y directly from raw splits
X_train = train_df.drop(columns=["price"])
y_train = train_df["price"]
# ... (similar for val/test)

# 2. Complete pipeline (feature engineering + scaling)
full_pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", numeric_preprocessor)
])

# 3. Fit on training data (learns min_date_)
full_pipeline.fit(X_train, y_train)

# 4. Transform all splits using fitted pipeline
X_train_processed = full_pipeline.transform(X_train)
X_val_processed = full_pipeline.transform(X_val)   # Uses training min_date_ ✓
X_test_processed = full_pipeline.transform(X_test) # Uses training min_date_ ✓

# 5. Save COMPLETE pipeline
joblib.dump(full_pipeline, "preprocessor.joblib")
```

**Benefits**:
- ✅ No data leakage - consistent `min_date_` across splits
- ✅ Complete saved pipeline - ready for inference
- ✅ Automatic state management - `min_date_` in pipeline
- ✅ Composable - standard sklearn interface

---

### Feature Value Comparison

Let's trace a specific example with actual dates:

**Setup**:
- Training dates: 2014-05-02 to 2015-02-15
- Validation dates: 2015-02-16 to 2015-04-01
- Test dates: 2015-04-02 to 2015-05-27

**House sold on 2015-03-01** (in validation set):

#### Before (INCORRECT)
```python
# Validation set processing
val_eng = engineer_features(val_df)
# Inside engineer_features:
#   min_date = val_df["date_parsed"].min() = 2015-02-16
#   days_since_start = (2015-03-01) - (2015-02-16) = 13 days

# Result: days_since_start = 13
```

#### After (CORRECT)
```python
# Pipeline fitted on training
full_pipeline.fit(X_train)
# Fitted: min_date_ = 2014-05-02

# Transform validation
X_val_processed = full_pipeline.transform(X_val)
# Inside transform:
#   days_since_start = (2015-03-01) - (2014-05-02) = 303 days

# Result: days_since_start = 303 ✓
```

**Impact on model**: The model trained on training data expects `days_since_start` values around 200-300 for late-period sales. The "After" version provides consistent values, while "Before" would confuse the model with artificially low values.

---

### Inference Comparison

#### Before (FAILS)

```python
# Load saved pipeline
preprocessor = joblib.load("preprocessor.joblib")

# New data arrives
new_df = pd.read_csv("new_houses_2015.csv")
new_df["date_parsed"] = pd.to_datetime(new_df["date"].str[:8])

# Attempt inference
try:
    predictions = model.predict(preprocessor.transform(new_df))
except KeyError as e:
    print(f"Error: {e}")
    # KeyError: "['days_since_start', 'house_age', ...] not found"
```

**Why it fails**: The saved pipeline expects engineered features that don't exist in raw data.

#### After (WORKS)

```python
# Load saved pipeline (includes feature engineering)
full_pipeline = joblib.load("preprocessor.joblib")

# New data arrives
new_df = pd.read_csv("new_houses_2015.csv")
new_df["date_parsed"] = pd.to_datetime(new_df["date"].str[:8])

# Inference works end-to-end
X_new = new_df.drop(columns=["price"])
predictions = model.predict(full_pipeline.transform(X_new))
# ✓ Pipeline creates features using fitted min_date_ from training
```

**Why it works**: The complete pipeline includes feature engineering with fitted reference values.

---

## Technical Deep Dive: sklearn Transformer Protocol

### BaseEstimator and TransformerMixin

```python
class FeatureEngineer(BaseEstimator, TransformerMixin):
```

#### BaseEstimator
Provides:
- `get_params()`: Returns initialization parameters
- `set_params()`: Sets initialization parameters
- Enables `GridSearchCV` and `RandomizedSearchCV` integration

#### TransformerMixin
Provides:
- `fit_transform()`: Calls `fit()` then `transform()` efficiently
- Standard transformer interface

### Method Signatures

```python
def fit(self, X: pd.DataFrame, y=None):
    """
    Must return self for method chaining.
    y parameter required even if unused (sklearn convention).
    """
    self.min_date_ = X["date_parsed"].min()
    return self  # ← Critical for Pipeline compatibility

def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    """
    Must return transformed data.
    Should not modify X in-place (hence X.copy()).
    """
    X = X.copy()  # ← Prevent side effects
    # ... transformations ...
    return X
```

### Pipeline Execution Flow

```python
full_pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", numeric_preprocessor)
])

# When calling: full_pipeline.fit(X_train, y_train)
# Step 1: feature_engineering.fit(X_train, y_train)
#         → Stores min_date_ = 2014-05-02
#         → Returns self
# Step 2: X_temp = feature_engineering.transform(X_train)
#         → Applies feature engineering
# Step 3: preprocessing.fit(X_temp, y_train)
#         → Fits scalers on engineered features
#         → Returns self

# When calling: full_pipeline.transform(X_val)
# Step 1: X_temp = feature_engineering.transform(X_val)
#         → Uses fitted min_date_ = 2014-05-02
# Step 2: X_final = preprocessing.transform(X_temp)
#         → Uses fitted scaling parameters
```

---

## Verification and Testing

### Verify Consistent Feature Values

```python
# Fit pipeline on training
full_pipeline.fit(X_train, y_train)

# Check fitted reference
print(f"Fitted min_date: {full_pipeline.named_steps['feature_engineering'].min_date_.date()}")
# Output: Fitted min_date: 2014-05-02

# Transform all splits
X_train_t = full_pipeline.transform(X_train)
X_val_t = full_pipeline.transform(X_val)
X_test_t = full_pipeline.transform(X_test)

# Verify days_since_start ranges are consistent
# (val/test should have HIGHER values than train, not reset to 0)
print(f"Train days_since_start range: {X_train_t[:, feature_idx].min():.0f} to {X_train_t[:, feature_idx].max():.0f}")
print(f"Val   days_since_start range: {X_val_t[:, feature_idx].min():.0f} to {X_val_t[:, feature_idx].max():.0f}")
print(f"Test  days_since_start range: {X_test_t[:, feature_idx].min():.0f} to {X_test_t[:, feature_idx].max():.0f}")

# Expected output (correct):
# Train: 0 to 289
# Val:   290 to 335  ← Higher than train ✓
# Test:  336 to 390  ← Higher than val ✓

# NOT (incorrect):
# Train: 0 to 289
# Val:   0 to 45     ← Reset to 0 ✗
# Test:  0 to 55     ← Reset to 0 ✗
```

### Test End-to-End Inference

```python
# Save complete pipeline
joblib.dump(full_pipeline, "preprocessor.joblib")

# Simulate new session (load from disk)
loaded_pipeline = joblib.load("preprocessor.joblib")

# Create synthetic new data
new_data = pd.DataFrame({
    'id': [1000000],
    'date': ['20150615T000000'],
    'price': [500000],
    'bedrooms': [3],
    # ... all required columns
})

# Parse date (preprocessing step)
new_data['date_parsed'] = pd.to_datetime(new_data['date'].str[:8], format="%Y%m%d")

# Transform (should work without errors)
X_new = new_data.drop(columns=['price'])
X_new_processed = loaded_pipeline.transform(X_new)

# Verify days_since_start uses training reference
expected_days = (pd.Timestamp('2015-06-15') - pd.Timestamp('2014-05-02')).days
actual_days = X_new_processed[0, feature_idx]  # Assuming we know the feature index

assert abs(actual_days - expected_days) < 1, "days_since_start uses wrong reference!"
print(f"✓ Inference works correctly: days_since_start = {actual_days:.0f} (expected {expected_days})")
```

---

## Educational Value for Students

This refactoring demonstrates several important ML engineering concepts:

### 1. Data Leakage Subtlety

Students learn that data leakage can be **subtle and non-obvious**:
- Not just "using test labels during training"
- Can occur through improper reference value computation
- Requires careful thinking about `fit` vs `transform` semantics

### 2. sklearn Design Patterns

Students see **proper sklearn transformer design**:
- Separation of `fit()` (learn from training) and `transform()` (apply to any data)
- Fitted parameters convention (trailing underscore)
- Immutability (don't modify input data)
- Composability (works in `Pipeline`, `GridSearchCV`, etc.)

### 3. Production ML Considerations

Students understand **real-world ML deployment needs**:
- Saved models must be self-contained
- Inference should require minimal manual intervention
- Reproducibility requires capturing all transformation logic
- State management (reference values) must be explicit

### 4. Temporal Data Handling

Students learn **time-series specific challenges**:
- Temporal splits require special handling
- Date parsing must precede splitting
- Feature engineering must use training-only references
- Test set represents "future" data unseen during training

---

## Recommendations for Further Improvements

While the current implementation is correct, consider these enhancements:

### 1. Add Date Parsing to Pipeline

For true end-to-end inference from CSV:

```python
class DateParser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["date_parsed"] = pd.to_datetime(X["date"].str[:8], format="%Y%m%d")
        return X

# Extended pipeline
full_pipeline = Pipeline([
    ("date_parsing", DateParser()),
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", numeric_preprocessor)
])
```

### 2. Add Input Validation

```python
def fit(self, X: pd.DataFrame, y=None):
    # Validate required columns
    required_cols = ['date_parsed', 'yr_built', 'yr_renovated', 'sqft_basement', ...]
    missing = set(required_cols) - set(X.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    self.min_date_ = X["date_parsed"].min()
    return self
```

### 3. Handle Edge Cases

```python
def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    
    # Handle future dates beyond training range
    if X["date_parsed"].max() > (self.min_date_ + pd.Timedelta(days=730)):
        warnings.warn("Data contains dates >2 years beyond training range")
    
    # ... rest of transform ...
```

### 4. Add Unit Tests

```python
def test_feature_engineer_no_leakage():
    """Test that min_date_ is learned from training only."""
    train = pd.DataFrame({'date_parsed': pd.date_range('2014-01-01', periods=100)})
    test = pd.DataFrame({'date_parsed': pd.date_range('2015-01-01', periods=50)})
    
    fe = FeatureEngineer()
    fe.fit(train)
    
    # min_date should be from training, not test
    assert fe.min_date_ == pd.Timestamp('2014-01-01')
    
    # Transform test using training reference
    test_transformed = fe.transform(test)
    assert test_transformed['days_since_start'].min() == 365  # ~1 year later
```

---

## Conclusion

This refactoring addresses three critical issues:

1. **Fixed data leakage**: `days_since_start` now uses consistent training reference across all splits
2. **Enabled complete inference**: Saved pipeline includes all transformations with fitted parameters
3. **Adopted sklearn best practices**: Custom transformer follows standard conventions for production ML

The resulting code is:
- ✅ **Correct**: No data leakage
- ✅ **Reproducible**: Complete pipeline in single artifact
- ✅ **Maintainable**: Standard sklearn patterns
- ✅ **Production-ready**: End-to-end inference capability
- ✅ **Educational**: Demonstrates proper ML engineering

Students learning from this example will understand not just how to process data, but how to do so in a way that scales to production systems while maintaining scientific rigor.

---

## Appendix: Complete Code Comparison

### Before (INCORRECT)

```python
# Feature engineering function (outside pipeline)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # PROBLEM: Different min_date per split
    min_date = df["date_parsed"].min()
    df["days_since_start"] = (df["date_parsed"] - min_date).dt.days
    
    sale_year = df["date_parsed"].dt.year
    df["house_age"] = sale_year - df["yr_built"]
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["years_since_renovation"] = np.where(
        df["yr_renovated"] > 0, sale_year - df["yr_renovated"], 0)
    df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"].replace(0, 1)
    df["living_vs_neighbors"] = df["sqft_living"] / df["sqft_living15"].replace(0, 1)
    df["lot_vs_neighbors"] = df["sqft_lot"] / df["sqft_lot15"].replace(0, 1)
    
    columns_to_drop = ["id", "date", "date_parsed", "zipcode", "yr_built", "yr_renovated"]
    df = df.drop(columns=columns_to_drop)
    return df

# Apply separately (DATA LEAKAGE)
train_eng = engineer_features(train_df)
val_eng = engineer_features(val_df)
test_eng = engineer_features(test_df)

# Separate X/y
X_train = train_eng.drop(columns=["price"])
y_train = train_eng["price"]
# ...

# Partial pipeline (missing feature engineering)
preprocessor = ColumnTransformer([
    ("log", log_pipeline, log_features),
    ("scale", StandardScaler(), scale_features),
    ("passthrough", "passthrough", passthrough_features)
])

preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# INCOMPLETE pipeline saved
joblib.dump(preprocessor, "preprocessor.joblib")
```

### After (CORRECT)

```python
# Feature engineering as sklearn transformer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X: pd.DataFrame, y=None):
        # Learn min_date from TRAINING data only
        self.min_date_ = X["date_parsed"].min()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Use FITTED min_date_ for all splits (no leakage)
        X["days_since_start"] = (X["date_parsed"] - self.min_date_).dt.days
        
        sale_year = X["date_parsed"].dt.year
        X["house_age"] = sale_year - X["yr_built"]
        X["was_renovated"] = (X["yr_renovated"] > 0).astype(int)
        X["years_since_renovation"] = np.where(
            X["yr_renovated"] > 0, sale_year - X["yr_renovated"], 0)
        X["basement_ratio"] = X["sqft_basement"] / X["sqft_living"].replace(0, 1)
        X["living_vs_neighbors"] = X["sqft_living"] / X["sqft_living15"].replace(0, 1)
        X["lot_vs_neighbors"] = X["sqft_lot"] / X["sqft_lot15"].replace(0, 1)
        
        columns_to_drop = ["id", "date", "date_parsed", "zipcode", "yr_built", "yr_renovated"]
        X = X.drop(columns=columns_to_drop)
        return X

# Separate X/y from raw splits
X_train = train_df.drop(columns=["price"])
y_train = train_df["price"]
# ...

# Numeric preprocessing
numeric_preprocessor = ColumnTransformer([
    ("log", log_pipeline, log_features),
    ("scale", StandardScaler(), scale_features),
    ("passthrough", "passthrough", passthrough_features)
])

# Complete pipeline (feature engineering + preprocessing)
full_pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", numeric_preprocessor)
])

# Fit on training (learns min_date_)
full_pipeline.fit(X_train, y_train)

# Transform all splits with consistent reference
X_train_processed = full_pipeline.transform(X_train)
X_val_processed = full_pipeline.transform(X_val)
X_test_processed = full_pipeline.transform(X_test)

# COMPLETE pipeline saved (includes feature engineering + fitted min_date_)
joblib.dump(full_pipeline, "preprocessor.joblib")
```

---

**Document Version**: 1.0  
**Date**: February 3, 2026  
**Author**: AI Assistant (GitHub Copilot)  
**Related Files**: 
- [04-preprocessing.ipynb](04-preprocessing.ipynb)
- `processed_data/preprocessor.joblib`
