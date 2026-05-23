import pandas as pd

def load_data(path_to_train: str, path_to_test: str, target_col: str, id_col: str):
    df_train = pd.read_csv(path_to_train)
    df_test  = pd.read_csv(path_to_test)

    train_ids = df_train[id_col].copy()
    test_ids  = df_test[id_col].copy()

    y_train = df_train[target_col].copy()
    
    # Удаляем и таргет, и ID из матрицы признаков
    X_train = df_train.drop(columns=[target_col, id_col]).copy()
    X_test  = df_test.drop(columns=[id_col]).copy()

    return X_train, y_train, X_test, train_ids, test_ids