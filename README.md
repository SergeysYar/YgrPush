# Shampoo Quality Forecast

Система прогнозирования итогового качества шампуня по данным технологических этапов варки.

Проект собирает данные из производственной SQLite-базы, строит признаки по этапам процесса, обучает ML-модели и предоставляет интерфейсы для анализа, обучения и предсказаний через CLI, API и dashboard.

## Документация

Для подробного и пошагового знакомства с системой начните с хаба документации:

- `docs/README.md`

Быстрые ссылки по разделам:

- `docs/01-overview.md` — основная идея, цели и ценность системы
- `docs/00-quick-start.md` — быстрый запуск и первый рабочий сценарий
- `docs/02-business-scenario.md` — как система применяется в производственном процессе
- `docs/03-architecture.md` — архитектура проекта и зоны ответственности модулей
- `docs/04-data-and-database.md` — устройство БД, поток данных, очистка, таргеты, датасеты
- `docs/05-ml-and-training.md` — модели, обучение, валидация, champion/challenger
- `docs/06-api.md` — API, сценарии вызова и структура ответов
- `docs/07-cli.md` — CLI-команды и операционные сценарии
- `docs/08-dashboard.md` — структура и назначение dashboard
- `docs/09-operations.md` — артефакты, хранение, переобучение, сопровождение
- `docs/11-faq.md` — частые вопросы по смыслу, данным и эксплуатации
- `docs/12-api-examples.md` — быстрые примеры вызовов API
- `docs/13-diagrams.md` — визуальные схемы архитектуры и потока данных
- `docs/10-limitations-and-roadmap.md` — ограничения текущей версии и roadmap

## Назначение

Система помогает заранее оценить качество партии до получения итогового лабораторного протокола. Основные прогнозируемые показатели:

- итоговый `pH`
- итоговая `вязкость`
- итоговое `содержание хлоридов`

Прогноз носит рекомендательный характер и поддерживает технолога в принятии решений, но не заменяет лабораторный контроль.

## Что уже реализовано

- чтение производственной SQLite-базы в режиме read-only
- извлечение batch-, product- и protocol-данных
- очистка и нормализация измерений
- batch-level и snapshot-level feature engineering
- построение тренировочных датасетов без утечки будущих этапов
- базовые ML-модели: `Baseline`, `Ridge`, `PLS`, `BayesianRidge`
- групповая валидация `LeaveOneGroupOut` по `batch_id`
- хранение моделей и результатов в `ml_storage.db`
- FastAPI endpoints для обучения, предикта, отчётов и обзора системы
- CLI-команды для эксплуатации и локальной разработки
- заготовка dashboard для визуального анализа

## Архитектура

Ключевые каталоги проекта:

- `app/config.py` — конфигурация через `.env`
- `app/database/` — доступ к SQLite и хранение ML-артефактов
- `app/data/` — очистка данных, таргеты, сборка датасетов
- `app/features/` — генерация batch- и snapshot-признаков
- `app/ml/` — модели, валидация, pipeline и prediction service
- `app/api/` — FastAPI-приложение
- `app/dashboard/` — Streamlit dashboard
- `tests/` — тесты ключевых модулей
- `docs/` — подробная пользовательская и техническая документация

## Поток данных

1. Система читает `production.db` только на чтение.
2. Производственные измерения очищаются и приводятся к рабочему формату.
3. Для партии подтягиваются связанные batch-, product- и protocol-данные.
4. Из лабораторных таблиц извлекаются целевые показатели качества.
5. Строятся итоговые и промежуточные признаки по этапам процесса.
6. Модели обучаются и сохраняются в `ml_storage.db` и `artifacts/`.
7. CLI, API и dashboard используют сохранённые модели для прогноза и анализа.

## Требования

- Python `3.12`
- доступ к `production.db`
- зависимости из `pyproject.toml`

## Установка

```bash
python -m pip install --upgrade pip
pip install -e .
```

После установки:

1. Скопируйте `.env.example` в `.env`
2. Проверьте пути к БД и служебным файлам
3. При необходимости скорректируйте конфигурацию под локальное окружение

## Конфигурация

Основные параметры:

```env
DB_PATH=./production.db
ML_STORAGE_PATH=./ml_storage.db
TARGET_PROTOCOL_POLICY=latest
CONFIDENCE_HIGH_MIN_BATCHES=20
```

## CLI

Основные команды:

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

## Быстрый старт

```bash
python -m app.cli validate-data
python -m app.cli train
python -m app.cli run-api
```

После этого можно:

- открыть API и проверить endpoints
- выполнить предсказание по конкретной партии
- перейти в `docs/README.md` для детального изучения системы

## Статус

Текущая версия — рабочий MVP с уже собранным контуром данных, обучения и предсказания. Следующие улучшения и оставшиеся блоки подробно описаны в `docs/10-limitations-and-roadmap.md`.
