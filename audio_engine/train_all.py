"""
Retrains the 5 models of the audio engine with ONE command.

    venv\\Scripts\\python audio_engine\\train_all.py

The notebooks stay the training code (with their plots and their MLflow runs): this script only runs them
headless, in order, and saves their outputs (the plots stay visible in the notebooks):
  02 SVM -> 03 decision tree -> 04 CNN -> 06 ResNet50 / VGG16 -> 07 Transformer-CNN -> 05 benchmark + ONNX export
(04 must run before 06 and 07: it makes the spectrograms they use. 05 runs last: it reads the MLflow runs of all the
models and writes the ONNX files + benchmark.json of this folder.)
Before running: notebook 01 made splits/test_files.json, and the MLflow server runs on http://localhost:5000.
"""
import os
import subprocess
import sys
import time

NOTEBOOKS = [
    "02_Tp3_svm.ipynb",
    "03_Tp4_decision_tree.ipynb",
    "04_Tp7_cnn.ipynb",
    "06_Tp8_transfer_learning.ipynb",
    "07_transformer_cnn.ipynb",
    "05_best_model_onnx.ipynb",
]
NOTEBOOK_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")

for notebook in NOTEBOOKS:
    path = os.path.join(NOTEBOOK_FOLDER, notebook)
    print("==========", notebook, "...", flush=True)
    start = time.time()
    command = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               "--ExecutePreprocessor.timeout=-1", path]
    subprocess.run(command, check=True)    # check=True: stop at the first notebook that fails
    print("done in {:.1f} min".format((time.time() - start) / 60), flush=True)

print("All the models are trained, logged in MLflow and exported to ONNX (audio_engine/).")
