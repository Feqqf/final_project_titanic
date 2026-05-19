""" Загрузка данных. Разделеление на фичи и таргет, выделение id  """

import pandas as pd
from config import config


def load_data(path_to_train:str, path_to_test:str, target_col:str, id_col:str):
    """Принимает пути к тренировочному датасету и тестовому датасету, а также 
        принимает целевую переменную и новый id;

        Вернуть необходимо тренировочную выборку с фичами и таргетом, и тестовую выборку.
        Вернуть id для пассаижров для тренировочной и тестовой выборки

        X_train, y_train, X_test : матрицы признаков и вектор целевой переменной
        train_ids, test_ids : идентификаторы пассажиров
         """
    df_train = pd.read_csv(path_to_train)
    df_test = pd.read_csv(path_to_test)

    train_ids = df_train[id_col].copy()
    test_ids = df_test[id_col].copy()

    X_train = df_train.drop(target_col, axis=1).copy()
    y_train = df_train[target_col].copy()
    X_test = df_test.copy()

    return X_train, y_train, X_test, train_ids, test_ids


