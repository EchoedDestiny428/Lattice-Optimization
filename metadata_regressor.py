import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# ============================================================
# STAGE 1: LOAD AND CLEAN DATASET
# ============================================================
CSV_PATH = "data/dataset_compiled.csv"

if not os.path.exists(CSV_PATH):
    print(f"Error: Could not find {CSV_PATH}. Run your batch simulation first!")
    exit()

print("\n[Stage 1/4] Extracting and preparing data metrics...")
with tqdm(total=3, desc="Data Pipeline", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
    df = pd.read_csv(CSV_PATH)
    pbar.update(1)
    
    df_clean = df[(df["status"] == "SUCCESS") & (df["reaction_force_fz_n"] < 0)]
    pbar.update(1)
    
    feature_cols = ["actual_density", "threshold", "freq", "noise"]
    target_col = "E_eff_gpa"
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pbar.update(1)

print(f"-> Ready. Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ============================================================
# STAGE 2: CONFIGURING ESTIMATORS (With Custom Progress Bar Hook)
# ============================================================
print("\n[Stage 2/4] Initializing Gradient Boosting hyperparameters...")
N_TREES = 200

class TqdmXGBCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_trees):
        super().__init__()
        self.pbar = tqdm(total=total_trees, desc="Training Model", unit="tree")
        
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

model = xgb.XGBRegressor(
    n_estimators=N_TREES,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    callbacks=[TqdmXGBCallback(N_TREES)]
)

# ============================================================
# STAGE 3: TRAINING TREE REGRESSORS 
# ============================================================
print(f"\n[Stage 3/4] Training {N_TREES} structural decision trees...")

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False  # Hide the messy default text logs
)

print("-> Target fitting optimized successfully.")

# ============================================================
# STAGE 4: MODEL EVALUATION
# ============================================================
print("\n[Stage 4/4] Validating statistical constraints...")
with tqdm(total=2, desc="Evaluating", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
    y_pred = model.predict(X_test)
    pbar.update(1)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pbar.update(1)

print("\n==============================================")
print("            MODEL EVALUATION METRICS          ")
print("==============================================")
print(f"R² Score (Accuracy):   {r2 * 100:.2f}%")
print(f"RMSE Error:            {rmse:.5f} GPa")
print("==============================================")

print("\nSample Comparisons (Actual vs Predicted):")
comparison_df = pd.DataFrame({"Actual GPa": y_test, "Predicted GPa": y_pred})
print(comparison_df.head(10).to_string(index=False))