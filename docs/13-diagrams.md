# Диаграммы системы

Этот раздел помогает быстро понять устройство системы визуально. Диаграммы сделаны в формате Mermaid, чтобы их можно было поддерживать прямо в репозитории.

## Общая архитектура

```mermaid
flowchart LR
    DB[production.db] --> DATA[Data Layer]
    DATA --> FEAT[Feature Engineering]
    FEAT --> TRAIN[Training Pipeline]
    TRAIN --> STORE[ml_storage.db / artifacts]
    STORE --> API[FastAPI]
    STORE --> CLI[CLI]
    STORE --> DASH[Dashboard]
```

Смысл диаграммы:

- производственная база является источником фактов
- data layer очищает и подготавливает данные
- feature engineering собирает признаки
- training pipeline создаёт модели
- сохранённые модели используются всеми интерфейсами доступа

## Поток данных от партии к прогнозу

```mermaid
flowchart TD
    A[Batch in production.db] --> B[Measurements extraction]
    B --> C[Cleaning and normalization]
    C --> D[Batch and snapshot features]
    D --> E[Model inference]
    E --> F[Prediction result]
    F --> G[CLI / API / Dashboard]
```

Эта схема показывает рабочий путь данных в момент прогноза.

## Жизненный цикл модели

```mermaid
flowchart TD
    A[Extract history] --> B[Build dataset]
    B --> C[Train candidate models]
    C --> D[Validate by batch groups]
    D --> E[Select best model]
    E --> F[Save metadata and artifact]
    F --> G[Promote active model]
    G --> H[Use in predictions]
    H --> I[Compare with actual values]
```

Эта диаграмма полезна для понимания MLOps-контура системы даже в MVP-формате.

## Архитектура модулей проекта

```mermaid
flowchart TB
    CFG[app/config.py]
    DBL[app/database]
    DAT[app/data]
    FEA[app/features]
    ML[app/ml]
    API[app/api]
    CLI[app/cli]
    DSH[app/dashboard]

    CFG --> DBL
    CFG --> DAT
    CFG --> ML
    DBL --> DAT
    DAT --> FEA
    FEA --> ML
    ML --> API
    ML --> CLI
    ML --> DSH
```

## Когда диаграммы особенно полезны

- при онбординге нового участника команды
- при обсуждении архитектуры с бизнесом или технологами
- при подготовке презентации проекта
- при планировании следующих этапов развития
