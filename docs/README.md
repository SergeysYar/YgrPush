# Документация системы

Этот раздел собран как подробный путеводитель по системе прогнозирования качества шампуня.

Если вы открыли проект впервые, рекомендуемый порядок чтения такой:

1. `docs/01-overview.md`
2. `docs/02-business-scenario.md`
3. `docs/03-architecture.md`
4. `docs/04-data-and-database.md`
5. `docs/05-ml-and-training.md`
6. `docs/06-api.md`
7. `docs/07-cli.md`
8. `docs/08-dashboard.md`
9. `docs/09-operations.md`
10. `docs/10-limitations-and-roadmap.md`

## Карта документов

- `docs/01-overview.md` — что делает система, для кого она нужна и какая у неё главная идея
- `docs/02-business-scenario.md` — как система используется в производственном процессе
- `docs/03-architecture.md` — устройство проекта по слоям и ключевым модулям
- `docs/04-data-and-database.md` — откуда берутся данные, как они очищаются и превращаются в признаки
- `docs/05-ml-and-training.md` — обучение, валидация, модели, champion/challenger, CatBoost
- `docs/06-api.md` — подробное описание API и типовых сценариев использования
- `docs/07-cli.md` — команды CLI и ожидаемое поведение
- `docs/08-dashboard.md` — как устроен Streamlit-интерфейс
- `docs/09-operations.md` — эксплуатация, артефакты, переобучение, хранение результатов
- `docs/10-limitations-and-roadmap.md` — ограничения текущего MVP и направления развития

## Кому что читать

- **Технологу / бизнес-пользователю**
  - `docs/01-overview.md`
  - `docs/02-business-scenario.md`
  - `docs/08-dashboard.md`

- **Аналитику / ML-инженеру**
  - `docs/04-data-and-database.md`
  - `docs/05-ml-and-training.md`
  - `docs/10-limitations-and-roadmap.md`

- **Backend / platform-инженеру**
  - `docs/03-architecture.md`
  - `docs/06-api.md`
  - `docs/07-cli.md`
  - `docs/09-operations.md`

- **Новому участнику команды**
  - Начните с `docs/01-overview.md`, затем `docs/03-architecture.md`, потом `docs/06-api.md`
