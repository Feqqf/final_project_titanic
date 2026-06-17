"""
Конструирование признаков. Создаёт новые колонки из существующих сырых данных.
Все преобразования детерминированы и не требуют обученного состояния.
"""

import pandas as pd
import numpy as np 

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт Initial, Family_Size и Alone из сырых колонок.
    Вызывается на объединённом фрейме train+test, чтобы кодировка была единой.
    """
    df = df.copy()

    # ПРИЗНАК 1: Обращение пассажира (Initial)

    # Извлекаем звание в обществе из поля Name (например "Mr.", "Mrs.", "Miss, e.t.c")
    df['Initial'] = df.Name.str.extract(r'([A-Za-z]+)\.')

    # Редкие титулы схлопываем в четыре основные группы
    df['Initial'] = df['Initial'].replace({
        'Mlle': 'Miss',
        'Mme':  'Miss',
        'Ms':   'Miss',
        'Dr':   'Mr',
        'Major':'Mr',
        'Lady': 'Mrs',
        'Countess': 'Mrs',
        'Capt': 'Mr',
        'Sir':  'Mr',
        'Don':  'Mr',
    })

    df.loc[~df['Initial'].isin(['Mr', 'Miss', 'Mrs']), 'Initial'] = 'Other'

    # ПРИЗНАК 2: Размер семьи
    # Общее число родственников на борту (родители/дети + братья/сёстры/супруг)
    df['Family_Size'] = df['Parch'] + df['SibSp']
    
    # ПРИЗНАК 3: Статус одиночки
    # Бинарный флаг: 1 — путешествует один, 0 — с семьёй
    df['Alone'] = np.where(df['Family_Size'] == 0, 1, 0)

    return df