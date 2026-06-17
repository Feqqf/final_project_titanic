import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Определяем доступное устройство один раз глобально или внутри класса
def get_device():
    """
    Определяет наиболее подходящее устройство для выполнения вычислений PyTorch.

    Приоритет:
        1. CUDA (если доступна) – для NVIDIA GPU.
        2. MPS (если доступен) – для чипов Apple Silicon.
        3. CPU – в противном случае.

    Returns:
        torch.device: Выбранное устройство.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class ImprovedTitanicMLP(nn.Module):
    """
    Многослойный персептрон для задачи классификации (бинарной).

    Архитектура:
        - Входной слой (размер input_dim)
        - Скрытые слои с заданными размерами, активацией, BatchNorm (опционально) и Dropout
        - Выходной слой на 2 нейрона (для CrossEntropyLoss)

    Args:
        input_dim (int): Размерность входных признаков.
        hidden_dims (list[int]): Список размеров скрытых слоёв.
        dropout (float): Вероятность дропаута (0..1).
        use_batchnorm (bool): Применять ли BatchNorm после каждого скрытого слоя.
        activation (str): Тип активации: "relu" или "gelu".
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, 
                 use_batchnorm: bool = False, activation: str = "relu"):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Выбор функции активации
        act_fn = nn.ReLU() if activation == "relu" else nn.GELU()

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Для бинарной классификации в PyTorch с CrossEntropyLoss выходной слой должен иметь 2 нейрона
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.

        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, input_dim).

        Returns:
            torch.Tensor: Логиты (не нормализованные вероятности) формы (batch_size, 2).
        """
        return self.network(x)


class ImprovedTitanicNet(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn-совместимый классификатор на основе MLP с автоматическим выбором устройства
    и ранней остановкой.

    Параметры:
        hidden_dims (list[int] | None): Размеры скрытых слоёв. По умолчанию [64, 32].
        dropout (float): Вероятность дропаута (0..1). По умолчанию 0.2.
        lr (float): Скорость обучения (learning rate). По умолчанию 1e-3.
        epochs (int): Максимальное число эпох. По умолчанию 200.
        batch_size (int): Размер мини-батча. По умолчанию 32.
        patience (int): Число эпох без улучшения для ранней остановки. По умолчанию 15.
        val_fraction (float): Доля данных для валидации (0..1). По умолчанию 0.15.
        use_batchnorm (bool): Использовать ли BatchNorm. По умолчанию False (для малых данных).
        activation (str): Функция активации: "relu" или "gelu". По умолчанию "gelu".
        random_state (int): Seed для воспроизводимости. По умолчанию 42.
    """
    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 15,
        val_fraction: float = 0.15,
        use_batchnorm: bool = False, # Выключен по умолчанию для малых данных
        activation: str = "gelu",    # GELU часто работает лучше ReLU
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.random_state = random_state
        self.device = get_device() # Автоматический выбор устройства

    def fit(self, X, y):
        """
        Обучает модель на предоставленных данных с использованием ранней остановки.

        Процесс:
            1. Фиксирует random seed.
            2. Масштабирует признаки через StandardScaler.
            3. Разбивает данные на тренировочную и валидационную выборки (стратифицированно).
            4. Инициализирует MLP и переносит на выбранное устройство.
            5. Обучает модель с оптимизатором AdamW и функцией потерь CrossEntropyLoss.
            6. Отслеживает валидационные потери и точность; применяет раннюю остановку
               (сохраняет лучшую модель по комбинации потерь и точности).
            7. Восстанавливает лучшие веса после завершения обучения.

        Args:
            X (array-like): Признаки, форма (n_samples, n_features).
            y (array-like): Целевые метки, форма (n_samples,).

        Returns:
            self: Обученный классификатор.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        self.classes_ = np.unique(y)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=self.val_fraction, 
            random_state=self.random_state, stratify=y
        )

        # Инициализация модели и перенос на устройство
        self.model_ = ImprovedTitanicMLP(
            input_dim=X_train.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_batchnorm=self.use_batchnorm,
            activation=self.activation,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Перенос тензоров на устройство
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)
        X_val_t   = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t   = torch.tensor(y_val, dtype=torch.long, device=self.device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # --- Обучение ---
            self.model_.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            # --- Валидация ---
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                
                # Вычисляем accuracy для ранней остановки
                val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)

            # Логирование (можно убрать для тишины)
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:03d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Улучшенная ранняя остановка: сохраняем, если loss меньше ИЛИ (loss такой же, но acc выше)
            if val_loss < best_val_loss or (np.isclose(val_loss, best_val_loss) and val_acc > best_val_acc):
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Возвращает вероятности принадлежности к каждому классу.

        Args:
            X (array-like): Признаки, форма (n_samples, n_features).

        Returns:
            np.ndarray: Массив вероятностей формы (n_samples, n_classes). 
                Для бинарной классификации – (n_samples, 2).
        """
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        self.model_.eval()
        with torch.no_grad():
            # softmax уже встроен в CrossEntropyLoss для обучения, но для predict_proba он нужен
            proba = torch.softmax(self.model_(X_t), dim=1).cpu().numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        """
        Возвращает предсказанные классы для выборки X.

        Args:
            X (array-like): Признаки, форма (n_samples, n_features).

        Returns:
            np.ndarray: Массив предсказанных меток классов.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]