# Документация системы

Этот раздел — единая точка входа в документацию по системе прогнозирования качества шампуня. Здесь собраны материалы для разных ролей: бизнес-пользователей, технологов, аналитиков, ML-инженеров, backend-разработчиков и новых участников команды.

Если вы открыли проект впервые, лучше идти по следующему порядку:

1. `docs/00-quick-start.md`
2. `docs/01-overview.md`
3. `docs/02-business-scenario.md`
4. `docs/03-architecture.md`
5. `docs/04-data-and-database.md`
6. `docs/05-ml-and-training.md`
7. `docs/06-api.md`
8. `docs/07-cli.md`
9. `docs/08-dashboard.md`
10. `docs/09-operations.md`
11. `docs/11-faq.md`
12. `docs/12-api-examples.md`
13. `docs/13-diagrams.md`
14. `docs/14-scientific-foundation.md`
15. `docs/15-experiments-and-reproducibility.md`
16. `docs/16-results-and-metrics.md`
17. `docs/17-methodology.md`
18. `docs/10-limitations-and-roadmap.md`

## Карта документов

- `docs/00-quick-start.md` — самый короткий путь от клона репозитория до первого запуска
- `docs/01-overview.md` — главная идея системы, цели и бизнес-ценность
- `docs/02-business-scenario.md` — как система встраивается в производственный процесс
- `docs/03-architecture.md` — устройство проекта по слоям, сервисам и модулям
- `docs/04-data-and-database.md` — источники данных, очистка, таргеты, признаки и датасеты
- `docs/05-ml-and-training.md` — обучение, валидация, модели, сохранение артефактов
- `docs/06-api.md` — описание API и контрактов на уровне маршрутов
- `docs/07-cli.md` — операционные команды и сценарии использования CLI
- `docs/08-dashboard.md` — назначение и структура dashboard
- `docs/09-operations.md` — сопровождение, хранение моделей, переобучение, контроль качества
- `docs/10-limitations-and-roadmap.md` — ограничения текущего MVP и направления развития
- `docs/11-faq.md` — ответы на частые вопросы по эксплуатации и смыслу системы
- `docs/12-api-examples.md` — готовые примеры вызовов API и ожидаемой логики работы
- `docs/13-diagrams.md` — Mermaid-диаграммы по потоку данных, архитектуре и жизненному циклу модели
- `docs/14-scientific-foundation.md` — научная постановка задачи, исследовательская ценность и сильные методологические стороны проекта
- `docs/15-experiments-and-reproducibility.md` — как в проекте устроены эксперименты, сравнение моделей и воспроизводимость результатов
- `docs/16-results-and-metrics.md` — как представлять результаты, какие метрики важны и как интерпретировать качество моделей
- `docs/17-methodology.md` — академичное описание методологии проекта: постановка, данные, датасет, валидация, моделирование и ограничения

## Сценарии чтения по ролям

### Технологу и бизнес-пользователю

Рекомендуемый порядок:

1. `docs/00-quick-start.md`
2. `docs/01-overview.md`
3. `docs/02-business-scenario.md`
4. `docs/08-dashboard.md`
5. `docs/11-faq.md`
6. `docs/14-scientific-foundation.md`
7. `docs/16-results-and-metrics.md`

### Аналитику и ML-инженеру

Рекомендуемый порядок:

1. `docs/00-quick-start.md`
2. `docs/03-architecture.md`
3. `docs/04-data-and-database.md`
4. `docs/05-ml-and-training.md`
5. `docs/09-operations.md`
6. `docs/10-limitations-and-roadmap.md`
7. `docs/15-experiments-and-reproducibility.md`
8. `docs/17-methodology.md`

### Backend- и platform-инженеру

Рекомендуемый порядок:

1. `docs/00-quick-start.md`
2. `docs/03-architecture.md`
3. `docs/06-api.md`
4. `docs/07-cli.md`
5. `docs/09-operations.md`
6. `docs/13-diagrams.md`
7. `docs/15-experiments-and-reproducibility.md`
8. `docs/17-methodology.md`

### Новому участнику команды

Рекомендуемый порядок:

1. `docs/00-quick-start.md`
2. `docs/01-overview.md`
3. `docs/03-architecture.md`
4. `docs/04-data-and-database.md`
5. `docs/06-api.md`
6. `docs/11-faq.md`
7. `docs/14-scientific-foundation.md`
8. `docs/17-methodology.md`

## Что читать в зависимости от задачи

- Нужно быстро понять, что вообще делает система — начните с `docs/01-overview.md`
- Нужно запустить проект локально — откройте `docs/00-quick-start.md`
- Нужно понять, откуда берутся прогнозы — смотрите `docs/04-data-and-database.md` и `docs/05-ml-and-training.md`
- Нужно встроить систему в другой контур — начните с `docs/06-api.md` и `docs/12-api-examples.md`
- Нужно сопровождать систему в проде — откройте `docs/09-operations.md`
- Нужно быстро ответить на типовой вопрос стейкхолдера — проверьте `docs/11-faq.md`
- Нужно показать, что проект имеет сильную научную и исследовательскую основу — начните с `docs/14-scientific-foundation.md`
- Нужно показать зрелость экспериментов и воспроизводимость — откройте `docs/15-experiments-and-reproducibility.md`
- Нужно показать результаты проекта как исследовательской работы — откройте `docs/16-results-and-metrics.md`
- Нужно описать проект в формате методологии или mini-paper — начните с `docs/17-methodology.md`

## Связь с главным README

Главная обзорная страница проекта находится в `README.md`. Она даёт короткое представление о системе, а текущий раздел раскрывает её подробно и по ролям.
