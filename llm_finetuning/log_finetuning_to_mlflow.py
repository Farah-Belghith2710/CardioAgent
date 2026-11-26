"""
Sends the results of the QLoRA fine-tuning (done on Google Colab) to MLflow (Service 6).

    venv\\Scripts\\python llm_finetuning\\log_finetuning_to_mlflow.py

Before running: download from Google Drive (MyDrive/cardioagent_llm/) into llm_finetuning/outputs/
  - trainer_state.json  (the loss of each training step, saved by the Colab notebook)
  - run_info.json       (the settings + the peak GPU memory)
  - cardio_lora_adapter.zip, unzipped in llm_finetuning/outputs/cardio_lora_adapter/
and put cardio-qwen2.5-7b-q4_k_m.gguf in llm_server/models/ (only its size and SHA256 go to MLflow: ~4.7 GB).

What goes to MLflow (experiment "llm_finetuning"):
  - params: base model, LoRA r / alpha, learning rate, epochs, max_seq_length, 4-bit, ...
  - metrics: train_loss (1 point per step -> MLflow draws the loss curve), learning_rate, peak_vram_gb, ...
  - artifacts: the loss curve (png), the adapter config, a sample of the dataset, the Modelfile, trainer_state.json
  - Model Registry: "cardio_llm" = the LoRA adapter (if MLflow accepts it)
"""
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")    # no window: the figure is only saved to a file
import matplotlib.pyplot as plt
import mlflow

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
ADAPTER_FOLDER = os.path.join(OUTPUTS, "cardio_lora_adapter")
GGUF_FILE = os.path.join(HERE, "..", "llm_server", "models", "cardio-qwen2.5-7b-q4_k_m.gguf")


def read_json(path):
    with open(path, encoding="utf-8-sig") as f:    # utf-8-sig: also reads a file saved with a BOM (Windows)
        return json.load(f)


def sha256_of(path):
    # the fingerprint of the GGUF file: proves which file the llm-server runs, without uploading 4.7 GB
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    state = read_json(os.path.join(OUTPUTS, "trainer_state.json"))
    info = read_json(os.path.join(OUTPUTS, "run_info.json"))

    # the training history: 1 line per logged step, e.g. {"step": 3, "loss": 1.82, "learning_rate": 0.0002, ...}
    steps = []
    losses = []
    for line in state["log_history"]:
        if "loss" in line:
            steps.append(line["step"])
            losses.append(line["loss"])
    print(len(steps), "training steps | first loss = {:.3f} | last loss = {:.3f}".format(losses[0], losses[-1]))

    # the loss curve as a figure (also drawn by MLflow from the metric)
    curve_file = os.path.join(OUTPUTS, "qlora_loss_curve.png")
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, marker=".")
    plt.title("QLoRA fine-tuning of " + info["base_model"] + " - training loss")
    plt.xlabel("step")
    plt.ylabel("training loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(curve_file)

    # a sample of the dataset (20 examples), to show in MLflow what the model learned from
    sample_file = os.path.join(OUTPUTS, "dataset_sample.jsonl")
    with open(os.path.join(HERE, "cardio_instructions.jsonl"), encoding="utf-8") as f:
        lines = f.readlines()
    with open(sample_file, "w", encoding="utf-8") as f:
        f.writelines(lines[:20])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("llm_finetuning")
    with mlflow.start_run(run_name="qlora_qwen2.5_7b") as run:
        mlflow.log_param("base_model", info["base_model"])
        mlflow.log_param("method", "QLoRA (Unsloth + TRL SFTTrainer)")
        mlflow.log_param("load_in_4bit", info["load_in_4bit"])
        mlflow.log_param("lora_r", info["lora_r"])
        mlflow.log_param("lora_alpha", info["lora_alpha"])
        mlflow.log_param("lora_target_modules", "q, k, v, o, gate, up, down")
        mlflow.log_param("learning_rate", info["learning_rate"])
        mlflow.log_param("epochs", info["epochs"])
        mlflow.log_param("max_seq_length", info["max_seq_length"])
        mlflow.log_param("batch_size", info["batch_size"])
        mlflow.log_param("gradient_accumulation", info["gradient_accumulation"])
        mlflow.log_param("n_examples", info["n_examples"])
        mlflow.log_param("gpu", info["gpu"])

        # the loss curve: 1 point per step
        for line in state["log_history"]:
            if "loss" in line:
                mlflow.log_metric("train_loss", line["loss"], step=line["step"])
                if "learning_rate" in line:
                    mlflow.log_metric("learning_rate", line["learning_rate"], step=line["step"])
                if "grad_norm" in line:
                    mlflow.log_metric("grad_norm", line["grad_norm"], step=line["step"])
        mlflow.log_metric("final_train_loss", losses[-1])
        mlflow.log_metric("peak_vram_gb", info["peak_vram_gb"])
        mlflow.log_metric("train_minutes", info["train_minutes"])

        mlflow.log_artifact(curve_file)
        mlflow.log_artifact(sample_file)
        mlflow.log_artifact(os.path.join(OUTPUTS, "trainer_state.json"))
        mlflow.log_artifact(os.path.join(HERE, "Modelfile"))

        # the LoRA adapter (small: a few dozen MB) -> artifact + Model Registry "cardio_llm"
        if os.path.exists(ADAPTER_FOLDER):
            mlflow.log_artifact(os.path.join(ADAPTER_FOLDER, "adapter_config.json"))
            mlflow.log_artifacts(ADAPTER_FOLDER, artifact_path="lora_adapter")
            try:
                mlflow.register_model("runs:/" + run.info.run_id + "/lora_adapter", "cardio_llm")
            except Exception as error:
                print("The Model Registry did not accept the adapter (it stays as an artifact of the run):", error)
        else:
            print("No adapter folder found in", ADAPTER_FOLDER, "-> unzip cardio_lora_adapter.zip there")

        # the GGUF file: too big to upload, so only its size and its SHA256 fingerprint
        if os.path.exists(GGUF_FILE):
            mlflow.log_param("gguf_file", os.path.basename(GGUF_FILE))
            mlflow.log_metric("gguf_size_gb", round(os.path.getsize(GGUF_FILE) / 1024**3, 2))
            mlflow.set_tag("gguf_sha256", sha256_of(GGUF_FILE))
        else:
            print("No GGUF file found in llm_server/models/ (its size and SHA256 are not logged)")

    print("Logged in MLflow, experiment llm_finetuning, run", run.info.run_id)


if __name__ == "__main__":
    main()
