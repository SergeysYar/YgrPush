from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.config import settings
from app.data.dataset_builder import DatasetBuilder
from app.database.ml_storage import ModelRegistry
from app.database.repositories import BatchRepository
from app.ml.prediction_service import PredictionService


st.set_page_config(page_title="Прогноз качества шампуня", layout="wide")


TARGET_LABELS = {
    "ph": "pH",
    "viscosity": "Вязкость",
    "chlorides": "Хлориды",
}

STATUS_LABELS = {
    "normal": "Норма",
    "risk": "Риск",
    "high_risk": "Высокий риск",
    "no_standard": "Нет норматива",
    "no_model_available": "Нет модели",
}

CONFIDENCE_LABELS = {
    "high": "Высокая",
    "medium": "Средняя",
    "low": "Низкая",
}


@st.cache_resource(show_spinner=False)
def get_services() -> dict[str, Any]:
    return {
        "builder": DatasetBuilder(settings.db_path),
        "batches": BatchRepository(settings.db_path),
        "models": ModelRegistry(settings.ml_storage_path),
        "predictions": PredictionService(settings.db_path, settings.ml_storage_path),
    }


def format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}".replace(",", " ")
    except (TypeError, ValueError):
        return str(value)


def status_badge(status: str | None) -> str:
    if not status:
        return "—"
    return STATUS_LABELS.get(status, status)


def confidence_badge(confidence: str | None) -> str:
    if not confidence:
        return "—"
    return CONFIDENCE_LABELS.get(confidence, confidence)


def build_batch_options(items: list[dict[str, Any]]) -> dict[str, int]:
    options: dict[str, int] = {}
    for item in items:
        batch_id = int(item["batch_id"])
        batch_number = item.get("batch_number") or "без номера"
        product_name = item.get("product_name") or "без продукта"
        options[f"{batch_id} · {batch_number} · {product_name}"] = batch_id
    return options


def render_overview(builder: DatasetBuilder, registry: ModelRegistry) -> None:
    summary = builder.get_api_summary()
    models = registry.list_models()
    new_batches = max(
        int(summary["total_batches"]) - int(summary["labeled_batches"]["ph"]),
        0,
    )

    st.subheader("Обзор данных")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Варок в базе", summary["total_batches"])
    metric_cols[1].metric("Размеченных по pH", summary["labeled_batches"]["ph"])
    metric_cols[2].metric("Новых варок", new_batches)
    metric_cols[3].metric("Пропусков", summary["missing_count"])
    metric_cols[4].metric("Доступных моделей", len(models))

    extra_cols = st.columns(3)
    extra_cols[0].metric("Лаб. протоколов", summary["protocol_count"])
    extra_cols[1].metric("Повторных протоколов", summary["batches_with_repeated_protocols"])
    extra_cols[2].metric("Подозрительных значений", summary["suspicious_count"])

    distribution = summary.get("step_distribution", {})
    if distribution:
        step_frame = pd.DataFrame(
            {
                "batch_id": list(distribution.keys()),
                "steps": list(distribution.values()),
            }
        )
        st.caption("Распределение количества выполненных этапов по варкам")
        st.bar_chart(step_frame.set_index("batch_id"))


def render_prediction(repository: BatchRepository, service: PredictionService) -> None:
    st.subheader("Прогноз варки")
    batches = repository.list_batches(limit=500, offset=0)
    if not batches:
        st.info("В базе нет доступных варок для прогноза.")
        return

    batch_options = build_batch_options(batches)
    selected_label = st.selectbox(
        "Выберите варку",
        options=list(batch_options.keys()),
    )
    batch_id = int(batch_options[selected_label])
    batch_details = repository.get_batch_detail(batch_id)
    measurements = batch_details["measurements"] if batch_details else []
    max_steps = max(len(measurements), 1)

    control_cols = st.columns([2, 1, 1])
    up_to_step = control_cols[0].slider(
        "Контрольный этап",
        min_value=1,
        max_value=max_steps,
        value=max_steps,
    )
    show_components = control_cols[1].toggle("Показать компоненты", value=False)
    show_measurements = control_cols[2].toggle("Показать этапы", value=False)

    if st.button("Сделать прогноз", type="primary", use_container_width=True):
        try:
            service.train_if_no_model()
            payload = service.predict_batch(batch_id, up_to_step_order=up_to_step)
        except Exception as error:
            st.error(f"Не удалось построить прогноз: {error}")
            return

        checkpoint = payload["checkpoint"]
        st.caption(
            f"Прогноз на {checkpoint['completed_steps']} завершённых этапах, "
            f"последний measurement_id: {checkpoint['last_measurement_id']}"
        )

        prediction_cols = st.columns(3)
        for idx, target in enumerate(("ph", "viscosity", "chlorides")):
            card = payload["predictions"][target]
            with prediction_cols[idx]:
                st.metric(TARGET_LABELS[target], format_number(card["value"]))
                st.write(f"Интервал: {format_number(card['lower'])} — {format_number(card['upper'])}")
                st.write(f"Статус: {status_badge(card['status'])}")
                st.write(f"Уверенность: {confidence_badge(card['confidence'])}")
                st.write(f"Обучающих варок: {card['training_batches']}")
                st.write(f"Модель: `{card['model'] or '—'}`")
                factors = card.get("top_factors") or []
                if factors:
                    st.write("Ключевые факторы:")
                    for factor in factors[:5]:
                        st.write(f"- {factor}")

        quality = payload["data_quality"]
        st.markdown("#### Качество данных текущего среза")
        quality_cols = st.columns(4)
        quality_cols[0].metric("Измерений", quality["total_measurements"])
        quality_cols[1].metric("С ошибками", quality["measurements_with_issues"])
        quality_cols[2].metric("Пропусков", quality.get("missing_count", 0))
        quality_cols[3].metric("Некорректных", quality.get("invalid_count", 0))

        similar_batches = payload.get("similar_batches") or []
        st.markdown("#### Похожие варки")
        if similar_batches:
            similar_frame = pd.DataFrame(similar_batches)
            st.dataframe(
                similar_frame.rename(
                    columns={
                        "batch_id": "batch_id",
                        "batch_number": "Номер варки",
                        "product_name": "Продукт",
                        "production_date": "Дата",
                        "distance": "Дистанция",
                        "ph": "pH",
                        "viscosity": "Вязкость",
                        "chlorides": "Хлориды",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Похожие варки пока не найдены.")

        if show_components and batch_details:
            st.markdown("#### Компоненты варки")
            components_frame = pd.DataFrame(batch_details["components"])
            if not components_frame.empty:
                st.dataframe(components_frame, use_container_width=True, hide_index=True)
            else:
                st.info("Для этой варки не найдено нормализованных компонентов.")

        if show_measurements and batch_details:
            st.markdown("#### Выполненные этапы")
            measurements_frame = pd.DataFrame(batch_details["measurements"])
            if not measurements_frame.empty:
                st.dataframe(measurements_frame.head(up_to_step), use_container_width=True, hide_index=True)


def render_models(registry: ModelRegistry) -> None:
    st.subheader("Модели")
    models = registry.list_models()
    if not models:
        st.info("Обученные модели пока отсутствуют.")
        return

    models_frame = pd.DataFrame(models)
    if "metrics_json" in models_frame.columns:
        models_frame["metrics_json"] = models_frame["metrics_json"].astype(str)
    if "features_json" in models_frame.columns:
        models_frame["features_json"] = models_frame["features_json"].astype(str)

    champions = models_frame[models_frame["status"] == "champion"]
    challengers = models_frame[models_frame["status"] == "challenger"]
    top_cols = st.columns(4)
    top_cols[0].metric("Всего моделей", len(models_frame))
    top_cols[1].metric("Champion", len(champions))
    top_cols[2].metric("Challenger", len(challengers))
    top_cols[3].metric("Целей покрытия", models_frame["target"].nunique())

    preferred_columns = [
        "model_id",
        "model_type",
        "target",
        "status",
        "created_at",
        "artifact_path",
        "metrics_json",
    ]
    visible_columns = [column for column in preferred_columns if column in models_frame.columns]
    st.dataframe(models_frame[visible_columns], use_container_width=True, hide_index=True)


def render_data_quality(builder: DatasetBuilder, repository: BatchRepository) -> None:
    st.subheader("Качество данных")
    summary = builder.get_data_summary()
    cols = st.columns(6)
    cols[0].metric("Измерений", summary["total_measurements"])
    cols[1].metric("Варок", summary["total_batches"])
    cols[2].metric("С целями", summary["batches_with_targets"])
    cols[3].metric("С ошибками", summary["measurements_with_issues"])
    cols[4].metric("Пропуски", summary["missing_count"])
    cols[5].metric("Out of range", summary["out_of_range_count"])

    issues_by_field = summary.get("issues_by_field") or {}
    if issues_by_field:
        issue_frame = pd.DataFrame(
            {"field": list(issues_by_field.keys()), "count": list(issues_by_field.values())}
        ).sort_values("count", ascending=False)
        st.caption("Проблемные поля")
        st.dataframe(issue_frame, use_container_width=True, hide_index=True)

    batches = repository.list_batches(limit=100, offset=0)
    if not batches:
        return

    selected_batch_id = st.selectbox(
        "Показать примеры проблем по варке",
        options=[int(item["batch_id"]) for item in batches],
        format_func=lambda batch_id: f"Варка {batch_id}",
        key="quality_batch_select",
    )
    details = repository.get_batch_detail(selected_batch_id)
    quality = details.get("data_quality", {}) if details else {}
    sample_issues = quality.get("sample_issues") or []
    if sample_issues:
        st.dataframe(pd.DataFrame(sample_issues), use_container_width=True, hide_index=True)
    else:
        st.success("Для выбранной варки примеры проблем не найдены.")


def main() -> None:
    services = get_services()
    builder: DatasetBuilder = services["builder"]
    batches: BatchRepository = services["batches"]
    models: ModelRegistry = services["models"]
    predictions: PredictionService = services["predictions"]

    st.title("Система прогнозирования качества шампуня")
    st.caption(
        "Инженерный MVP: обзор данных, прогноз по этапам варки, модели Champion/Challenger и контроль качества данных."
    )

    tabs = st.tabs(
        ["Обзор данных", "Прогноз варки", "Модели", "Качество данных"]
    )

    with tabs[0]:
        render_overview(builder, models)
    with tabs[1]:
        render_prediction(batches, predictions)
    with tabs[2]:
        render_models(models)
    with tabs[3]:
        render_data_quality(builder, batches)


if __name__ == "__main__":
    main()
