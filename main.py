import logging
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH, чтобы импорты работали из любой директории
sys.path.append(str(Path(__file__).parent))

from Config import config
from training import run_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    logging.info("Запуск пайплайна обучения")
    try:
        run_training_pipeline(config)
        logging.info("Пайплайн завершён успешно.")
    except FileNotFoundError as e:
        logging.error(f"Файл не найден: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Критическая ошибка пайплайна: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()