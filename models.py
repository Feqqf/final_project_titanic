from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def get_models(random_state: int = 42) -> dict:
    return {
        "dummy_classifier": DummyClassifier(strategy='most_frequent'),
        'Ridge (L2)': LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000),
        'Lasso (L1)': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000),
        'ElasticNet': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(random_state=random_state),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state),
        "lightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            min_child_samples=5,  # Фикс для маленького датасета
            random_state=random_state, verbose=-1  # Отключает предупреждения
        ),
        "catboost": CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state,
            eval_metric='logloss', verbosity=0  # Убран use_label_encoder
        )
    }

def get_model_by_name(model_name: str, random_state: int = 42):
    models = get_models(random_state=random_state)
    if model_name not in models:
        raise ValueError(f"Неизвестная модель '{model_name}'. Доступные: {', '.join(models.keys())}")
    return models[model_name]