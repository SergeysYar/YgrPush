FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY .env.example .env
RUN python -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
