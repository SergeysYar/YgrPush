# API

## Общая идея

API предназначен для трёх основных сценариев:

- получить данные и summary;
- обучить или просмотреть модели;
- запросить прогноз и отчёты.

Базовый префикс:

- `GET /health`
- `GET /api/v1/...`

## Основные маршруты

### Health

- `GET /health`

Проверяет доступность базы и ML storage.

### Data

- `GET /api/v1/data/summary`

Возвращает summary по данным:

- число измерений;
- число batch;
- число размеченных batch;
- число протоколов;
- распределение этапов и пр.

### Batches

- `GET /api/v1/batches`
- `GET /api/v1/batches/{batch_id}`

Сценарии:

- получить список batch;
- получить подробный payload по конкретной batch;
- использовать batch detail как точку входа для дальнейшего анализа.

### Train

- `POST /api/v1/train`

Позволяет:

- выбрать типы моделей;
- выбрать protocol policy;
- включить snapshot mode.

Пример:

```json
{
  "model_types": ["baseline", "ridge", "pls", "bayesian_ridge", "catboost"],
  "protocol_policy": "latest",
  "snapshot_mode": true
}
```

### Models

- `GET /api/v1/models`
- `GET /api/v1/models/summary`
- `GET /api/v1/models/runs`
- `POST /api/v1/models/{model_id}/promote`

Сценарии:

- посмотреть модели;
- посмотреть champion/challenger summary;
- посмотреть training runs;
- вручную промоутить модель.

### Predict

- `POST /api/v1/predict/batch/{batch_id}`
- `POST /api/v1/predict/custom`

Поддерживается:

- прогноз по batch из базы;
- прогноз по кастомному payload;
- ограничение по `up_to_step_order`;
- ограничение по `up_to_measurement_id`.

### Stored predictions

- `GET /api/v1/predictions`

Позволяет получить уже сохранённые прогнозы.

### Reports

- `GET /api/v1/reports/data-quality`
- `GET /api/v1/reports/data-quality/issues`

Первый маршрут умеет дополнительно сохранять issues:

- `GET /api/v1/reports/data-quality?store=true`

## Что возвращает prediction

Prediction response содержит:

- batch id;
- checkpoint;
- data quality summary;
- predictions по `ph`, `viscosity`, `chlorides`;
- similar batches.

Для каждого target:

- `value`
- `lower`
- `upper`
- `confidence`
- `model`
- `status`
- `training_batches`
- `top_factors`

## Ошибки

Типовые варианты:

- `404`, если batch не найдена;
- `400`, если custom payload некорректен;
- `500`, если произошла внутренняя ошибка сервиса.

## Практический совет

Если интеграция новая, обычно удобнее начинать с:

1. `GET /api/v1/data/summary`
2. `GET /api/v1/batches`
3. `POST /api/v1/predict/batch/{batch_id}`
4. `GET /api/v1/models/summary`
