"""
Предобработка данных. Заполняет пропуски, кодирует категории, бинирует непрерывные
признаки и удаляет колонки без сигнала.
"""

import pandas as pd
import numpy as np
from features import add_features


def drop_id_column(X_train: pd.DataFrame, X_test: pd.DataFrame, id_col: str):
    """Удаляет колонку ID — она не несёт предсказательной информации."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    if id_col in X_train.columns:
        X_train = X_train.drop(id_col, axis=1)
    if id_col in X_test.columns:
        X_test = X_test.drop(id_col, axis=1)

    return X_train, X_test


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет колонки, которые слишком разреженные, текстовые или уже
    закодированы в производных признаках (Age → Age_band, Fare → Fare_cat, Name → Initial).
    """
    df = df.copy()
    df = df.drop(['Cabin', 'Name', 'Age', 'Ticket', 'Fare'], axis=1)
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропуски групповыми средними, вычисленными на train.
    """
    df = df.copy()

    # Age: среднее по группе обращений (вычислено на train)
    df.loc[(df.Age.isnull()) & (df.Initial == 'Mr'),     'Age'] = 33
    df.loc[(df.Age.isnull()) & (df.Initial == 'Mrs'),    'Age'] = 36
    df.loc[(df.Age.isnull()) & (df.Initial == 'Master'), 'Age'] = 5
    df.loc[(df.Age.isnull()) & (df.Initial == 'Miss'),   'Age'] = 22
    df.loc[(df.Age.isnull()) & (df.Initial == 'Other'),  'Age'] = 46

    # Embarked: мода = S, всего 2 пропуска в train
    df['Embarked'] = df['Embarked'].fillna('S')

    # Fare: среднее по классу каюты, всего 1 пропуск в test
    df.loc[(df.Fare.isnull()) & (df.Pclass == 1), 'Fare'] = 94
    df.loc[(df.Fare.isnull()) & (df.Pclass == 2), 'Fare'] = 22
    df.loc[(df.Fare.isnull()) & (df.Pclass == 3), 'Fare'] = 12

    if np.sum(df.isna().sum()) == 0:
        print("Пропусков нет")
    else:
        print("Остались пропущенные значения — проверьте данные.")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Кодирует категории в целочисленные метки, непрерывные признаки — в порядковые бины.
    Label encoding достаточен: древесные модели доминируют, а бины сохраняют порядок
    для линейных.
    """
    df = df.copy()

    # --- Категории → целые числа ---
    df['Sex']      = df['Sex'].replace(['male', 'female'], [0, 1]).astype(int)
    df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2]).astype(int)
    df['Initial']  = df['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4]).astype(int)

    # --- Непрерывные → порядковые бины ---

    # Возрастные группы: ребёнок(0) / молодой(1) / взрослый(2) / средний(3) / старший(4)
    df['Age_band'] = 0
    df.loc[df['Age'] <= 16, 'Age_band'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age_band'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age_band'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age_band'] = 3
    df.loc[df['Age'] > 64, 'Age_band'] = 4

    # Ценовые категории на основе квантильных границ из EDA
    df['Fare_cat'] = 0
    df.loc[df['Fare'] <= 7.91, 'Fare_cat'] = 0
    df.loc[(df['Fare'] > 7.91)   & (df['Fare'] <= 14.454), 'Fare_cat'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31),     'Fare_cat'] = 2
    df.loc[(df['Fare'] > 31)     & (df['Fare'] <= 513),    'Fare_cat'] = 3

    return df


def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame, id_col: str):
    """
    Полная цепочка предобработки для train и test.

    Train и test конкатенируются перед трансформациями, чтобы конструирование
    признаков (например, извлечение обращений) работало одинаково для обоих наборов.
    После обработки они разделяются обратно по числу строк.
    """
    X_train, X_test = drop_id_column(X_train, X_test, id_col=id_col)

    train_rows = X_train.shape[0]

    # Объединяем, чтобы feature engineering применялся единообразно
    train_test_df = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

    train_test_df = add_basic_features(train_test_df)
    train_test_df = fill_missing_values(train_test_df)
    train_test_df = encode_features(train_test_df)
    train_test_df = drop_columns(train_test_df)

    X_train_processed = train_test_df.iloc[:train_rows, :].copy()
    X_test_processed  = train_test_df.iloc[train_rows:,  :].copy()

    # align() гарантирует, что test содержит ровно те же колонки в том же порядке, что и train;
    # пробелы заполняются 0 (безопасно, так как все признаки на этом этапе числовые)
    X_train_processed, X_test_processed = X_train_processed.align(
        X_test_processed,
        join="left",
        axis=1,
        fill_value=0,
    )

    return X_train_processed, X_test_processed


