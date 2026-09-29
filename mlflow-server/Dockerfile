FROM python:3.12-slim
WORKDIR /app

RUN pip install uv && uv pip install --system --no-cache-dir mlflow psycopg2-binary boto3

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "s3://mlflow-artifacts/", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
