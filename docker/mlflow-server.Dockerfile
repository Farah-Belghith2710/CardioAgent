# Service 6 : mlflow-server (MLflow Tracking + Model Registry)

# 1. start from the official MLflow image (same version as the mlflow library in the notebook)
FROM ghcr.io/mlflow/mlflow:v2.22.4

# 2. the folder where MLflow keeps everything: the database of the runs + the saved files (figures, models)
WORKDIR /mlflow

# 3. the MLflow web page listens on port 5000
EXPOSE 5000

# 4. command run when the container starts: the MLflow server
#    --backend-store-uri     : the runs, params and metrics go in a small database file (SQLite)
#    --artifacts-destination : the files (figures, models) go in the folder /mlflow/artifacts
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts"]