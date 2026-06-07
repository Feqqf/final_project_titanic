import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from models import get_models

def tune_model(model_name: str, X, y, cfg):
    """
    Запускает оптимизацию гиперпараметров для конкретной модели с помощью Optuna.
    """
    print(f"\nЗапуск тюнинга для модели: {model_name}")
    
    # 1. Получаем базовые параметры модели из конфига (чтобы не терять random_state и базовые настройки)
    base_models = get_models(cfg)
    base_model_instance = base_models[model_name]
    
    # Извлекаем класс модели и её текущие параметры (кроме random_state, его добавим отдельно)
    model_class = type(base_model_instance)
    base_params = base_model_instance.get_params()
    base_params.pop('random_state', None) 
    
    # 2. Получаем пространство поиска из конфига
    search_space = cfg.search_spaces.get(model_name, {})
    if not search_space:
        print(f"Для модели {model_name} не задан search_space в конфиге. Пропускаем.")
        return base_model_instance

    # 3. Функция цели (Objective) для Optuna
    def objective(trial):
        # Генерируем параметры для текущей попытки
        trial_params = base_params.copy()
        
        for param_name, space in search_space.items():
            if space['type'] == 'int':
                trial_params[param_name] = trial.suggest_int(
                    param_name, space['low'], space['high'], step=space.get('step', 1)
                )
            elif space['type'] == 'float':
                trial_params[param_name] = trial.suggest_float(
                    param_name, space['low'], space['high'], log=space.get('log', False)
                )
            elif space['type'] == 'categorical':
                trial_params[param_name] = trial.suggest_categorical(
                    param_name, space['choices']
                )
        
        # Добавляем random_state для воспроизводимости
        trial_params['random_state'] = cfg.cross_val_param.random_state
        
        # Создаем модель с новыми параметрами
        model = model_class(**trial_params)
        
        # Оцениваем с помощью кросс-валидации
        cv = StratifiedKFold(
            n_splits=cfg.cross_val_param.n_folds, 
            shuffle=True, 
            random_state=cfg.cross_val_param.random_state
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return scores.mean()

    # 4. Создаем и запускаем исследование (Study)
    # direction="maximize", так как мы хотим максимизировать accuracy
    study = optuna.create_study(direction="maximize")
    
    # Запускаем оптимизацию
    study.optimize(
        objective, 
        n_trials=cfg.tuning.n_trials, 
        timeout=cfg.tuning.get('timeout', None),
        show_progress_bar=True
    )

    print(f"Лучшая accuracy для {model_name}: {study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")

    # 5. Возвращаем модель, обученную на лучших параметрах
    best_params = {**base_params, **study.best_params, 'random_state': cfg.cross_val_param.random_state}
    return model_class(**best_params)