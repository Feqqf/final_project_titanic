"""
Список используемых моделей. 
Все модели определены здесь, чтобы добавить новую модель необходимо вписать ее в get_models().

"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def get_models(random_state: int = 42) -> dict:
    """
    Возвращает словарь {имя: необученный классификатор} для всех сравниваемых моделей.
    Каждый классификатор должен реализовывать fit / predict методы.
    """
    models = {
        # Baseline
        "dummy_classifier": DummyClassifier(strategy='most_frequent'),

        # Линейные(logreg)
        'Ridge (L2)': LogisticRegression(penalty='l2', solver='lbfgs',     C=1.0, max_iter=1000),
        'Lasso (L1)': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000),
        'ElasticNet': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=1000),

        # Метод k ближайших соседей и деревянные модели
        'KNN':           KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(random_state=random_state),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=101),

        # Градиентный бустинг
        "lightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=101, verbose=0),
        "catboost": CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=101),
        "XGBoost":  XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=101,
                                   use_label_encoder=False, eval_metric='logloss')
    }
    return models


def get_model_by_name(model_name: str, random_state: int = 42):
    """Возвращает свежий (необученный) экземпляр модели по имени."""
    models = get_models(random_state=random_state)

    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Неизвестная модель '{model_name}'. Доступные: {available}")

    return models[model_name]