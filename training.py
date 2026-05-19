"""
Оркестрация обучения. Координирует загрузку данных, предобработку, оценку моделей,
финальное обучение и сохранение всех результатов.
"""

import pandas as pd

from data import load_data
from evaluation import evaluate_models
from models import get_model_by_name, get_models
from preprocessing import preprocess_features



def run_training_pipeline(config):
    # --- Создаём выходные директории, если их нет ---
    # ensure_dir(config.output_dir)
    # ensure_dir(config.metrics_dir)
    # ensure_dir(config.models_dir)
    # ensure_dir(config.submissions_dir)
    # Дописать в конфиг пути коуда сохранять файлы и результаты

    # 1. Загрузка данных
    X_train, y_train, X_test, train_ids, test_ids = load_data(
        train_path=config.paths.path_to_train,
        test_path=config.paths.path_to_test,
        target_col=config.columns.target_col,
        id_col=config.columns.id_col,
    )

    # 2. Предобработка: конструирование признаков + кодирование (train и test вместе)
    X_train_processed, X_test_processed = preprocess_features(
        X_train=X_train,
        X_test=X_test,
        id_col=config.columns.id_col,
    )

    # 3. Оценка всех моделей через кросс-валидацию (тестовые данные не трогаются)
    models = get_models(random_state=config.cross_val_param.random_state)
    results_df = evaluate_models(
        models=models,
        X=X_train_processed,
        y=y_train,
        n_folds=config.cross_val_param.n_folds,
        random_state=config.cross_val_param.random_state,
    )

    best_model_name = results_df.iloc[0]["model_name"]
    best_cv_accuracy = float(results_df.iloc[0]["cv_accuracy_mean"])

    # 4. Выбор и финальное обучение на полной обучающей выборке
    # use_best_cv_model_for_final_fit=True  → берём лучшую по CV модель автоматически
    # use_best_cv_model_for_final_fit=False → берём config.main_model_name
    if config.choose_model.use_best_cv_model_for_final_fit:
        selected_model_name = best_model_name
    else:
        selected_model_name = config.choose_model.main_model_name

    final_model = get_model_by_name(
        model_name=selected_model_name,
        random_state=config.cross_val_param.random_state,
    )
    final_model.fit(X_train_processed, y_train)
    test_predictions = final_model.predict(X_test_processed)

    submission_df = pd.DataFrame({
        config.columns.id_col: test_ids,
        config.columns.target_col: test_predictions,
    })

    # 5. Сохранение результатов
    if config.save_metrics:
        save_dataframe(results_df, config.metrics_dir / "cv_results.csv")
        save_json(
            {
                "best_cv_model": best_model_name,
                "best_cv_accuracy": best_cv_accuracy,
                "selected_final_model": selected_model_name,
            },
            config.metrics_dir / "summary.json",
        )

    if config.save_submission:
        save_dataframe(
            submission_df,
            config.submissions_dir / f"submission_{selected_model_name}.csv",
        )

    if config.save_model:
        joblib.dump(
            final_model,
            config.models_dir / f"{selected_model_name}_model.joblib",
        )

    print("Обучение завершено.")
    print()
    print("Результаты кросс-валидации:")
    print(results_df)
    print()
    print(f"Лучшая модель по CV:    {best_model_name}")
    print(f"Выбранная модель:       {selected_model_name}")
    print(f"Сабмишн сохранён в:    {config.submissions_dir / f'submission_{selected_model_name}.csv'}")