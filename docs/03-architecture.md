# Архитектура системы

## Общая идея архитектуры

Проект построен слоями. Каждый слой отвечает за свою часть задачи:

- получение данных;
- очистку и подготовку;
- построение признаков;
- обучение и хранение моделей;
- выдачу прогнозов;
- пользовательский доступ через API, CLI и dashboard.

Такое разделение упрощает сопровождение: можно улучшать ML, API или отчётность независимо друг от друга.

## Структура проекта

### `app/config.py`

Центральная конфигурация приложения:

- путь к `production.db`
- путь к `ml_storage.db`
- политика выбора протокола
- пороги уверенности
- порог включения `CatBoost`

### `app/database/`

Слой работы с базами:

- чтение production SQLite;
- репозитории для batch/product/protocol данных;
- реестр моделей и хранилище артефактов;
- хранилище predictions и data quality issues.

Ключевые файлы:

- `app/database/source.py`
- `app/database/repositories.py`
- `app/database/ml_storage.py`

### `app/data/`

Слой подготовки данных:

- разбор чисел;
- очистка измерений;
- валидация;
- извлечение target-значений;
- построение batch- и snapshot-dataset.

Ключевые файлы:

- `app/data/numeric_parser.py`
- `app/data/cleaner.py`
- `app/data/validator.py`
- `app/data/target_loader.py`
- `app/data/dataset_builder.py`
- `app/data/data_quality.py`

### `app/features/`

Слой feature engineering:

- признаки по всей варке;
- признаки по промежуточным snapshot;
- логика checkpoint-представления.

Ключевые файлы:

- `app/features/batch_features.py`
- `app/features/snapshot_features.py`

### `app/ml/`

ML-слой:

- модели;
- кросс-валидация;
- обучение;
- реестр моделей;
- prediction service;
- экспорт отчётов.

Ключевые файлы:

- `app/ml/baseline.py`
- `app/ml/ridge.py`
- `app/ml/pls.py`
- `app/ml/bayesian.py`
- `app/ml/catboost_model.py`
- `app/ml/service.py`
- `app/ml/prediction_service.py`
- `app/ml/validation.py`
- `app/ml/registry.py`

### `app/api/`

HTTP-интерфейс:

- health
- data summary
- batches
- models
- training
- prediction
- reports

### `app/schemas/`

Pydantic-схемы API:

- batch-ответы
- prediction-ответы
- model-ответы
- report-ответы
- stored prediction-ответы

### `app/dashboard/`

Streamlit UI для человека:

- обзор данных
- прогноз по варке
- обзор моделей
- контроль качества данных

## Как слои взаимодействуют

Пример цепочки при запросе прогноза:

1. `app/api/routes/predict.py` принимает HTTP-запрос.
2. `PredictionService` загружает данные варки.
3. `DatasetBuilder` и репозитории собирают measurement и metadata.
4. `BatchFeatureBuilder` строит признаки.
5. Из реестра выбирается champion или fallback-модель.
6. Модель строит прогноз.
7. `PredictionService` формирует итоговый payload.
8. API возвращает типизированный ответ.

## Почему архитектура считается удачной для MVP

- она достаточно простая;
- понятны точки расширения;
- данные, ML и интерфейсы не перемешаны;
- возможна постепенная промышленная эволюция без полной переделки.
