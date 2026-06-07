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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class ImprovedTitanicMLP(nn.Module):
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
        return self.network(x)


class ImprovedTitanicNet(BaseEstimator, ClassifierMixin):
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
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        self.model_.eval()
        with torch.no_grad():
            # softmax уже встроен в CrossEntropyLoss для обучения, но для predict_proba он нужен
            proba = torch.softmax(self.model_(X_t), dim=1).cpu().numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]