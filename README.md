# Shampoo Quality Forecast

MVP системы прогнозирования конечного качества шампуня по данным технологических этапов варки.

Проект строит признаки по завершённым и промежуточным этапам варки, извлекает лабораторные целевые значения из `production.db`, обучает базовые ML-модели и предоставляет CLI, API и заготовку dashboard-интерфейса.

## Назначение

Система предназначена для предварительного прогноза показателей качества до получения лабораторного протокола:

- итоговый `pH`
- итоговая `вязкость`
- итоговое `содержание хлоридов`

Прогноз носит рекомендательный характер и не заменяет лабораторный контроль.

## Что уже реализовано

- чтение производственной SQLite-базы в режиме read-only
- извлечение batch-, product- и protocol-данных
- парсинг чисел с запятой и точкой
- базовая проверка качества данных
- batch-level и snapshot-level feature engineering
- snapshot-based training dataset без утечки будущих этапов
- `Baseline`, `Ridge`, `PLS`, `BayesianRidge`
- групповая валидация `LeaveOneGroupOut` по `batch_id`
- weighted training/validation для snapshot-режима
- хранение моделей и прогнозов в `ml_storage.db`
- FastAPI endpoints для summary, batches, train, predict, models, reports
- CLI-команды для инспекции БД, обучения, предикта и запуска API/dashboard

## Архитектура

- `app/config.py` — конфигурация через `.env`
- `app/database/` — доступ к SQLite и хранение ML-артефактов
- `app/data/` — очистка, целевые значения, dataset builder
- `app/features/` — batch и snapshot признаки
- `app/ml/` — модели, валидация, pipeline, prediction service
- `app/api/` — FastAPI
- `app/dashboard/` — Streamlit-заготовка
- `tests/` — unit/integration-style тесты для ключевых модулей

## Поток данных

1. `production.db` читается только для чтения.
2. Таблица `measurements` очищается и нормализуется.
3. Для batch'ей подтягиваются связанные данные из `Batches`, `Products`, `Loading_Process`, `Testing_Protocols`.
4. Из лабораторных таблиц извлекаются target-значения.
5. Строятся:
   - итоговые признаки по всей варке
   - промежуточные snapshot-признаки по мере выполнения этапов
6. Модели обучаются и сохраняются в `ml_storage.db` и `artifacts/`.
7. API и CLI используют обученные модели для предсказаний.

## Требования

- Python `3.12`
- SQLite база `production.db`

Основные зависимости описаны в `pyproject.toml`.

## Установка

```bash
python -m pip install --upgrade pip
pip install -e .
```

Затем скопируйте `.env.example` в `.env` и при необходимости поправьте пути.

## Конфигурация

Основные настройки:

```env
DB_PATH=./production.db
ML_STORAGE_PATH=./ml_storage.db
TARGET_PROTOCOL_POLICY=latest
CONFIDENCE_HIGH_MIN_BATCHES=20
```

## CLI

Доступные команды:

```bash
python -m app.cli inspect-db
python -m app.cli build-dataset
python -m app.cli validate-data
python -m app.cli train
python -m app.cli evaluate
python -m app.cli predict --batch-id 89
python -m app.cli predict --batch-id 89 --up-to-step 4
python -m app.cli list-models
python -m app.cli promote-model MODEL_ID
python -m app.cli update-actuals
python -m app.cli run-api
python -m app.cli run-dashboard
```

`train` и `evaluate` сейчас запускают pipeline в `snapshot_mode=True`.

## API

Основные маршруты:

- `GET /health`
- `GET /api/v1/data/summary`
- `GET /api/v1/batches`
- `GET /api/v1/batches/{batch_id}`
- `POST /api/v1/train`
- `GET /api/v1/models`
- `POST /api/v1/models/{model_id}/promote`
- `POST /api/v1/predict/batch/{batch_id}`
- `POST /api/v1/predict/custom`
- `GET /api/v1/predictions`
- `GET /api/v1/reports/data-quality`

### Пример обучения

```bash
curl -X POST http://127.0.0.1:8000/api/v1/train ^
  -H "Content-Type: application/json" ^
  -d "{\"model_types\":[\"baseline\",\"ridge\",\"pls\",\"bayesian_ridge\"],\"protocol_policy\":\"latest\",\"snapshot_mode\":true}"
```

### Пример предсказания

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict/batch/89?up_to_step_order=4"
```

Ответ включает:

- checkpoint-информацию
- data quality summary
- predictions по `ph`, `viscosity`, `chlorides`
- `status`
- `confidence`
- `training_batches`
- `top_factors`
- `similar_batches`

## Таблицы-источники

Используются следующие таблицы:

- `measurements`
- `Batches`
- `Products`
- `Loading_Process`
- `Testing_Protocols`
- `testing_protocol_values`
- `quality_targets`

## Модели

### Baseline

- среднее по обучающей выборке
- поддерживает weighted learning для snapshot-режима

### Ridge

- основной интерпретируемый baseline-уровень
- поддерживает объяснение top factors через вклады коэффициентов

### PLS

- многовыходная модель для полного набора целей
- пока используется в упрощённом режиме

### BayesianRidge

- даёт прогноз и используется для более узкого интервала, где возможно

## Валидация

Используется `LeaveOneGroupOut` по `batch_id`.

Это означает:

- все snapshots одной варки попадают только в train или только в test
- утечка информации между этапами одной варки исключается на уровне split-логики
- snapshot weights уменьшают дисбаланс длинных варок

Метрики:

- `MAE`
- `RMSE`
- `Median AE`
- `R²`

## Ограничения текущей реализации

На июль 2026 года проект реализован как рабочий инженерный MVP, но не все пункты исходного промышленного промта закрыты полностью.

Честные ограничения текущей версии:

- `CatBoost` и `SHAP` не доведены до production-ready состояния
- `Streamlit` пока остаётся заготовкой
- часть feature engineering всё ещё упрощена по сравнению с полным промтом
- `PLS` пока валидируется и используется в более простом режиме, чем задумывалось
- `top_factors` полноценно реализованы прежде всего для `Ridge`
- не все API-контракты из промта доведены до окончательной версии
- README описывает фактическую реализацию, а не идеальную целевую архитектуру

## Champion–Challenger

Модели сохраняются в `ml_storage.db`.

- новые модели получают статус `challenger`
- champion назначается вручную
- доступно через:

```bash
python -m app.cli promote-model MODEL_ID
```

## Docker

Сборка:

```bash
docker build -t shampoo-quality-forecast .
```

Запуск:

```bash
docker run --rm -p 8000:8000 shampoo-quality-forecast
```

Перед запуском убедитесь, что внутри контейнера доступен корректный `production.db`.

## Важное предупреждение

На текущем объёме данных прогноз носит исследовательский характер.
Надёжность должна подтверждаться на новых варках.

Модель не заменяет лабораторный контроль и не должна использоваться как единственный источник решения о качестве партии.
