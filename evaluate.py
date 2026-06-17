import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

def accuracy_cv(model, X, y, n_folds: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Выполняет кросс-валидацию с оценкой accuracy для заданной модели.

    Используется стратифицированный K-Fold разбиение, чтобы сохранить долю классов
    в каждом фолде (важно для несбалансированных данных).

    Args:
        model: Объект модели, совместимый с scikit-learn (имеет методы fit и predict).
        X (array-like): Матрица признаков, форма (n_samples, n_features).
        y (array-like): Вектор целевых меток, форма (n_samples,).
        n_folds (int): Количество фолдов для кросс-валидации. По умолчанию 5.
        random_state (int): Seed для перемешивания данных перед разбиением. По умолчанию 42.

    Returns:
        np.ndarray: Массив значений accuracy (доля правильных ответов) для каждого фолда,
            размер (n_folds,).
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return cross_val_score(model, X, y, scoring='accuracy', cv=cv)

def evaluate_models(models: dict, X, y, n_folds: int, random_state: int) -> pd.DataFrame:
    """
    Оценивает несколько моделей с помощью кросс-валидации и возвращает сводную таблицу.

    Для каждой модели из переданного словаря вычисляется средняя точность (accuracy)
    на основе кросс-валидации. Результаты сортируются по убыванию среднего значения.

    Args:
        models (dict): Словарь вида {имя_модели: объект_модели}. Модели должны иметь
            интерфейс scikit-learn.
        X (array-like): Матрица признаков, форма (n_samples, n_features).
        y (array-like): Вектор целевых меток, форма (n_samples,).
        n_folds (int): Количество фолдов для кросс-валидации.
        random_state (int): Seed для воспроизводимости разбиений.

    Returns:
        pd.DataFrame: Таблица с колонками:
            - 'model_name' – имя модели,
            - 'cv_accuracy_mean' – средняя accuracy по всем фолдам.
            Строки отсортированы по убыванию среднего accuracy.
    """
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