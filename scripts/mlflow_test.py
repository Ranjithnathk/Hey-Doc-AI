import mlflow

# Set experiment name (auto-creates folder under mlruns/)
mlflow.set_experiment("HeyDocAI_Summarizer")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("model", "facebook/bart-base")
    mlflow.log_metric("rouge1", 0.43)
    mlflow.log_metric("rougeL", 0.40)
    print("MLflow test run completed.")