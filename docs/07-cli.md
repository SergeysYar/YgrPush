# CLI

## Зачем нужен CLI

CLI нужен для инженерной и эксплуатационной работы:

- быстро проверить состояние данных;
- собрать dataset;
- обучить модели;
- сделать prediction;
- управлять champion/challenger;
- запустить API и dashboard.

## Основные команды

### Inspect database

```bash
python -m app.cli inspect-db
```

Показывает базовую информацию о структуре источника.

### Build dataset summary

```bash
python -m app.cli build-dataset
```

Возвращает summary по данным и качеству.

### Validate data

```bash
python -m app.cli validate-data
```

Показывает примеры проблем в измерениях.

### Train

```bash
python -m app.cli train
python -m app.cli train --model-types ridge bayesian_ridge
python -m app.cli train --protocol-policy first
python -m app.cli train --batch-mode
```

Поддерживает:

- выбор моделей;
- выбор protocol policy;
- snapshot или batch режим.

### Evaluate

```bash
python -m app.cli evaluate
```

В текущем проекте использует тот же training pipeline и сохраняет отчёт.

### Predict

```bash
python -m app.cli predict --batch-id 89
python -m app.cli predict --batch-id 89 --up-to-step 4
```

### List models

```bash
python -m app.cli list-models
```

### Promote model

```bash
python -m app.cli promote-model MODEL_ID
```

### Update actuals

```bash
python -m app.cli update-actuals
```

Обновляет фактические значения для ранее сохранённых predictions.

### Run API

```bash
python -m app.cli run-api
```

### Run dashboard

```bash
python -m app.cli run-dashboard
```

## Когда использовать CLI, а когда API

- **CLI** удобнее для разработчика, аналитика и оператора сопровождения
- **API** удобнее для внешней интеграции и UI

## Практический минимальный сценарий

Если нужно быстро начать:

1. `python -m app.cli inspect-db`
2. `python -m app.cli build-dataset`
3. `python -m app.cli train`
4. `python -m app.cli predict --batch-id <ID>`
