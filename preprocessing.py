import pandas as pd
import numpy as np
from features import add_features

def drop_id_column(X_train: pd.DataFrame, X_test: pd.DataFrame, id_col: str):
    """
    Удаляет столбец-идентификатор из обучающей и тестовой выборок.

    Args:
        X_train (pd.DataFrame): Обучающая выборка.
        X_test (pd.DataFrame): Тестовая выборка.
        id_col (str): Название столбца, который нужно удалить.

    Returns:
        tuple: (X_train, X_test) без столбца id_col.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    if id_col in X_train.columns:
        X_train = X_train.drop(id_col, axis=1)
    if id_col in X_test.columns:
        X_test = X_test.drop(id_col, axis=1)
    return X_train, X_test

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет из датафрейма столбцы, которые не используются в финальной модели.

    Среди удаляемых:
        - 'Cabin', 'Name', 'Ticket' – содержат много пропусков или уникальных значений;
        - 'Age' и 'Fare' – удаляются, потому что вместо них используются бинаризованные
          версии ('Age_band' и 'Fare_cat'), созданные в функции encode_features.

    Args:
        df (pd.DataFrame): Исходный датафрейм.

    Returns:
        pd.DataFrame: Датафрейм без указанных столбцов.
    """
    df = df.copy()
    cols_to_drop = ['Cabin', 'Name', 'Age', 'Ticket', 'Fare']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски в полях 'Age', 'Embarked' и 'Fare' осмысленными значениями.

    Для 'Age' используется медианный возраст по обращению (Initial):
        - Mr  → 33
        - Mrs → 36
        - Master → 5
        - Miss → 22
        - Other → 46
    Для 'Embarked' – наиболее частое значение 'S'.
    Для 'Fare' – медиана по классу каюты (Pclass).

    Args:
        df (pd.DataFrame): Датафрейм с пропусками.

    Returns:
        pd.DataFrame: Датафрейм с заполненными пропусками.
    """
    df = df.copy()
    df.loc[(df.Age.isnull()) & (df.Initial == 'Mr'),     'Age'] = 33
    df.loc[(df.Age.isnull()) & (df.Initial == 'Mrs'),    'Age'] = 36
    df.loc[(df.Age.isnull()) & (df.Initial == 'Master'), 'Age'] = 5
    df.loc[(df.Age.isnull()) & (df.Initial == 'Miss'),   'Age'] = 22
    df.loc[(df.Age.isnull()) & (df.Initial == 'Other'),  'Age'] = 46

    df['Embarked'] = df['Embarked'].fillna('S')
    df.loc[(df.Fare.isnull()) & (df.Pclass == 1), 'Fare'] = 94
    df.loc[(df.Fare.isnull()) & (df.Pclass == 2), 'Fare'] = 22
    df.loc[(df.Fare.isnull()) & (df.Pclass == 3), 'Fare'] = 12
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Кодирует категориальные признаки в числовой формат и создаёт бинаризованные версии
    для 'Age' и 'Fare'.

    Преобразования:
        - 'Sex'       : male→0, female→1
        - 'Embarked'  : S→0, C→1, Q→2
        - 'Initial'   : Mr→0, Mrs→1, Miss→2, Master→3, Other→4
        - 'Age_band'  : 5 возрастных групп (0–4)
        - 'Fare_cat'  : 4 группы на основе квантилей

    Args:
        df (pd.DataFrame): Датафрейм с заполненными пропусками.

    Returns:
        pd.DataFrame: Датафрейм с закодированными признаками.
    """
    df = df.copy()
    df['Sex']      = df['Sex'].replace(['male', 'female'], [0, 1]).astype(int)
    df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2]).astype(int)
    df['Initial']  = df['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4]).astype(int)

    df['Age_band'] = 0
    df.loc[df['Age']  <= 16, 'Age_band'] = 0
    df.loc[(df['Age']  > 16)  & (df['Age']  <= 32), 'Age_band'] = 1
    df.loc[(df['Age']  > 32)  & (df['Age']  <= 48), 'Age_band'] = 2
    df.loc[(df['Age']  > 48)  & (df['Age']  <= 64), 'Age_band'] = 3
    df.loc[df['Age']  > 64, 'Age_band'] = 4

    df['Fare_cat'] = 0
    df.loc[df['Fare']  <= 7.91, 'Fare_cat'] = 0
    df.loc[(df['Fare']  > 7.91)    & (df['Fare']  <= 14.454), 'Fare_cat'] = 1
    df.loc[(df['Fare']  > 14.454)  & (df['Fare']  <= 31),     'Fare_cat'] = 2
    df.loc[(df['Fare']  > 31)      & (df['Fare']  <= 513),    'Fare_cat'] = 3
    return df

def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame, id_col: str):
    """
    Основной пайплайн предобработки признаков для задачи Titanic.

    Шаги:
        1. Удаление столбца-идентификатора.
        2. Объединение train и test для единообразного применения преобразований.
        3. Добавление новых признаков через внешнюю функцию add_features.
        4. Заполнение пропусков (fill_missing_values).
        5. Кодирование категорий и создание бинов (encode_features).
        6. Удаление неинформативных столбцов (drop_columns).
        7. Разделение обратно на train и test.
        8. Выравнивание колонок (на случай, если в тесте отсутствуют редкие категории).

    Args:
        X_train (pd.DataFrame): Обучающая выборка (без целевой переменной).
        X_test (pd.DataFrame): Тестовая выборка.
        id_col (str): Имя столбца-идентификатора, который нужно удалить.

    Returns:
        tuple: (X_train_processed, X_test_processed) — обработанные данные.
    """
    X_train, X_test = drop_id_column(X_train, X_test, id_col)
    train_rows = X_train.shape[0]

    train_test_df = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    train_test_df = add_features(train_test_df)
    train_test_df = fill_missing_values(train_test_df)
    train_test_df = encode_features(train_test_df)
    train_test_df = drop_columns(train_test_df)

    X_train_processed = train_test_df.iloc[:train_rows, :].copy()
    X_test_processed  = train_test_df.iloc[train_rows:,  :].copy()

    # Выравниваем колонки (на случай, если в test появились/пропали редкие категории)
    X_train_processed, X_test_processed = X_train_processed.align(
        X_test_processed, join="left", axis=1, fill_value=0
    )
    return X_train_processed, X_test_processed


