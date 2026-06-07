from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.resolve()

config_dict = {
    'paths': {
        'path_to_train': str(PROJECT_ROOT / 'datas' / 'train.csv'),
        'path_to_test':  str(PROJECT_ROOT / 'datas' / 'test.csv'),
        'output_dir':    str(PROJECT_ROOT / 'output'),
    },
    'columns': {
        'target_col': 'Survived',
        'id_col': 'PassengerId'
    },
    'cross_val_param': {
        'n_folds': 5,
        'random_state': 42  # Единая точка контроля случайности
    },
    'choose_model': {
        'baseline_model': 'dummy_classifier',
        'main_model_name': 'XGBoost',
        'use_best_cv_model_for_final_fit': True
    },
    'flags': {
        'save_metrics': True,
        'save_model': True,
        'save_submission': True
    },
    # ПАРАМЕТРЫ ВСЕХ МОДЕЛЕЙ
    'models': {
        'dummy': {'strategy': 'most_frequent'},
        'ridge': {'l1_ratio': 0, 'solver': 'lbfgs', 'C': 1.0, 'max_iter': 1000},
        'lasso': {'l1_ratio': 1, 'solver': 'liblinear', 'C': 1.0, 'max_iter': 1000},
        'elastic_net': {'l1_ratio': 0.5, 'solver': 'saga', 'C': 1.0, 'max_iter': 1000},
        'knn': {}, 
        'decision_tree': {'min_samples_leaf': 3, 'max_features': 3},
        'random_forest': {'n_estimators': 300, 'max_depth': 5, 'min_samples_leaf': 3, 'max_features': 3},
        'lightgbm': {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 4, 'min_child_samples': 5, 'verbose': -1},
        'catboost': {'iterations': 300, 'learning_rate': 0.01, 'depth': 4, 'verbose': 0},
        'xgboost': {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 4, 'eval_metric': 'logloss', 'verbosity': 0},
        'neural_net': {
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3,
            'lr': 1e-3,
            'epochs': 150,
            'batch_size': 16,
            'patience': 20,
            'use_batchnorm': False,
            'activation': "relu"
        }
    },
    # СЕКЦИИ ДЛЯ OPTUNA
    'tuning': {
        'enabled': True,          # Переключатель: True - тюним, False - берем дефолтные из 'models'
        'n_trials': 20,           # Количество попыток (для начала хватит 20-30)
        'timeout': 300,           # Максимальное время на тюнинг одной модели в секундах (опционально)
        'models_to_tune': ['XGBoost', 'random_forest','catboost','NeuralNet'] # Какие модели будем улучшать
    },
    'search_spaces': {
        'XGBoost': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 500, 'step': 50},
            'max_depth': {'type': 'int', 'low': 3, 'high': 8},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'min_child_weight': {'type': 'int', 'low': 1, 'high': 10}
        },
        'catboost':{
            'iterations':{'type': 'int', 'low': 100, 'high': 500, 'step': 50},
            'depth': {'type': 'int', 'low': 3, 'high': 8},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high':0.3, 'log': True}
        },
        'random_forest': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 500, 'step': 50},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5}
        },
        'NeuralNet': {
            'hidden_dims': {'type': 'categorical', 'choices': [[64, 32], [128, 64], [128, 64, 32]]},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]}
        }
    }
}

config = OmegaConf.create(config_dict)

_out = Path(config.paths.output_dir)
config.paths.metrics_dir = str(_out / 'metrics')
config.paths.models_dir = str(_out / 'models')
config.paths.submissions_dir = str(_out / 'submissions')