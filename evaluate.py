import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

def accuracy_cv(model, X, y, n_folds: int = 5, random_state: int = 42) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return cross_val_score(model, X, y, scoring='accuracy', cv=cv)

def evaluate_models(models: dict, X, y, n_folds: int, random_state: int) -> pd.DataFrame:
    results = []
    for model_name, model in models.items():
        scores = accuracy_cv(model, X, y, n_folds, random_state)
        results.append({
            "model_name": model_name,
            "cv_accuracy_mean": scores.mean(),
        })

    return (
        pd.DataFrame(results)
        .sort_values("cv_accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )