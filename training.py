import os
import json
import joblib
import pandas as pd
from pathlib import Path

from data import load_data
from evaluate import evaluate_models
from models import get_model_by_name, get_models
from preprocessing import preprocess_features
from Config import config

# Импортируем тюнинг условно, чтобы не ломать код, если optuna не установлен (хотя мы его установили)
try:
    from tuning import tune_model
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

def ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_dataframe(df: pd.DataFrame, path: str):
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"DataFrame saved: {path}")

def save_json(data: dict, path: str):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"JSON saved: {path}")

def run_training_pipeline(cfg=config):
    print("Загрузка данных")
    X_train, y_train, X_test, train_ids, test_ids = load_data(
        path_to_train=cfg.paths.path_to_train,
        path_to_test=cfg.paths.path_to_test,
        target_col=cfg.columns.target_col,
        id_col=cfg.columns.id_col,
    )

    print("Предобработка")
    X_train_processed, X_test_processed = preprocess_features(
        X_train=X_train,
        X_test=X_test,
        id_col=cfg.columns.id_col,
    )

    print("Инициализация базовых моделей")
    models = get_models(cfg)
    
    # OPTUNA TUNING BLOCK
    
    if cfg.tuning.enabled and OPTUNA_AVAILABLE:
        print("\nЗапуск подбора гиперпараметров (Optuna)...")
        for model_name in cfg.tuning.models_to_tune:
            if model_name in models:
                # Заменяем базовую модель на модель с лучшими найденными параметрами
                models[model_name] = tune_model(model_name, X_train_processed, y_train, cfg)
            else:
                print(f"Модель '{model_name}' не найдена в get_models(). Пропускаем тюнинг.")
    elif cfg.tuning.enabled and not OPTUNA_AVAILABLE:
        print("Тюнинг включен в конфиге, но библиотека optuna не найдена. Пропускаем.")
    

    print("\nФинальная кросс-валидация моделей")
    results_df = evaluate_models(
        models=models,
        X=X_train_processed,
        y=y_train,
        n_folds=cfg.cross_val_param.n_folds,
        random_state=cfg.cross_val_param.random_state,
    )

    best_model_name = results_df.iloc[0]["model_name"]
    best_cv_accuracy = float(results_df.iloc[0]["cv_accuracy_mean"])

    print("\nФинальное обучение")
    if cfg.choose_model.use_best_cv_model_for_final_fit:
        selected_model_name = best_model_name
    else:
        selected_model_name = cfg.choose_model.main_model_name

    final_model = get_model_by_name(
        model_name=selected_model_name,
        cfg=cfg
    )
    
    # Обучаем финальную модель на ВСЕХ тренировочных данных
    final_model.fit(X_train_processed, y_train)
    test_predictions = final_model.predict(X_test_processed)

    submission_df = pd.DataFrame({
        cfg.columns.id_col: test_ids,
        cfg.columns.target_col: test_predictions,
    })

    print("\nСохранение результатов")
    if cfg.flags.save_metrics:
        save_dataframe(results_df, os.path.join(cfg.paths.metrics_dir, "cv_results.csv"))
        save_json({
            "best_cv_model": best_model_name,
            "best_cv_accuracy": best_cv_accuracy,
            "selected_final_model": selected_model_name,
        }, os.path.join(cfg.paths.metrics_dir, "summary.json"))

    if cfg.flags.save_submission:
        save_dataframe(
            submission_df,
            os.path.join(cfg.paths.submissions_dir, f"submission_{selected_model_name}.csv"),
        )

    if cfg.flags.save_model:
        ensure_dir(os.path.join(cfg.paths.models_dir, "dummy")) # На случай, если папки нет
        joblib.dump(
            final_model,
            os.path.join(cfg.paths.models_dir, f"{selected_model_name}_model.joblib"),
        )

    print("\nОбучение завершено.")
    print("Результаты кросс-валидации:")
    print(results_df)
    print(f"Лучшая модель по CV:    {best_model_name}")
    print(f"Выбранная модель:       {selected_model_name}")