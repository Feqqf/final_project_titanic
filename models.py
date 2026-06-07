from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from NeuralNet import ImprovedTitanicNet
from config import config as cfg


def get_models(cfg) -> dict:
    """
    Фабрика моделей. Читает все гиперпараметры из конфига.
    """
    rs = cfg.cross_val_param.random_state
    
    # Вспомогательная функция: берет конфиг модели и добавляет туда random_state
    # Мы делаем dict(), чтобы преобразовать OmegaConf DictConfig в обычный словарь Python
    def with_rs(model_cfg):
        params = dict(model_cfg)
        params['random_state'] = rs
        return params

    return {
        "dummy_classifier": DummyClassifier(**cfg.models.dummy),
        'Ridge (L2)': LogisticRegression(**with_rs(cfg.models.ridge)),
        'Lasso (L1)': LogisticRegression(**with_rs(cfg.models.lasso)),
        'ElasticNet': LogisticRegression(**with_rs(cfg.models.elastic_net)),
        'KNN': KNeighborsClassifier(**cfg.models.knn),
        'decision_tree': DecisionTreeClassifier(**with_rs(cfg.models.decision_tree)),
        'random_forest': RandomForestClassifier(**with_rs(cfg.models.random_forest)),
        "lightGBM": LGBMClassifier(**with_rs(cfg.models.lightgbm)),
        "catboost": CatBoostClassifier(**with_rs(cfg.models.catboost)),
        "XGBoost": XGBClassifier(**with_rs(cfg.models.xgboost)),
        'NeuralNet': ImprovedTitanicNet(
            random_state=rs,
            **cfg.models.neural_net  # Распаковываем все параметры нейросети
        )
    }


def get_model_by_name(model_name: str, cfg):
    """
    Получает конкретную модель по имени, используя общий конфиг.
    """
    models = get_models(cfg)
    if model_name not in models:
        raise ValueError(f"Неизвестная модель '{model_name}'. Доступные: {', '.join(models.keys())}")
    return models[model_name]