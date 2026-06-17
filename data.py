import pandas as pd

def load_data(path_to_train: str, path_to_test: str, target_col: str, id_col: str):
    """
    Загружает тренировочные и тестовые данные из CSV-файлов,
    разделяет их на матрицу признаков (X), целевую переменную (y) и ID.

    Args:
        path_to_train (str): Путь к тренировочному датасету.
        path_to_test (str): Путь к тестовому датасету.
        target_col (str): Название целевой переменной (таргета).
        id_col (str): Название столбца с идентификаторами.

    Returns:
        tuple: Кортеж из 5 элементов:
            - X_train (pd.DataFrame): Признаки тренировочной выборки.
            - y_train (pd.Series): Таргет тренировочной выборки.
            - X_test (pd.DataFrame): Признаки тестовой выборки.
            - train_ids (pd.Series): ID тренировочной выборки.
            - test_ids (pd.Series): ID тестовой выборки.
    """
    df_train = pd.read_csv(path_to_train)
    df_test  = pd.read_csv(path_to_test)

    train_ids = df_train[id_col].copy()
    test_ids  = df_test[id_col].copy()

    y_train = df_train[target_col].copy()
    
    # Удаляем и таргет, и ID из матрицы признаков
    X_train = df_train.drop(columns=[target_col, id_col]).copy()
    X_test  = df_test.drop(columns=[id_col]).copy()

    return X_train, y_train, X_test, train_ids, test_ids