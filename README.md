# Kidney Disease Classification (MLflow + DVC)

A deep-learning project that classifies kidney CT scan slices as **Normal** or **Tumor**. It uses transfer learning on a VGG16 network pretrained on ImageNet. The training pipeline is reproducible with DVC, MLflow can track the runs, a Flask web app lets anyone upload a scan and check it, and GitHub Actions builds a Docker image and deploys it to AWS (ECR + EC2).

> **Not a medical device.** This project is for research and learning. Its predictions are not a diagnosis. Anyone with a health concern should talk to a doctor.

## Contents

- [Results](#results)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Dataset](#dataset)
- [Training pipeline (DVC)](#training-pipeline-dvc)
- [Configuration](#configuration)
- [Model](#model)
- [Experiment tracking (MLflow)](#experiment-tracking-mlflow)
- [Web app](#web-app)
- [REST API](#rest-api)
- [Docker](#docker)
- [CI/CD on AWS with GitHub Actions](#cicd-on-aws-with-github-actions)
- [Development workflow](#development-workflow)
- [Limitations](#limitations)
- [License and credits](#license-and-credits)

## Results

Scores from the current [scores.json](scores.json). They come from 1,104 held-out test slices that the model never saw during training, with **Tumor** treated as the positive class.

| Metric | Value | What it means |
|---|---|---|
| Accuracy | **87.0%** | share of all test slices classified correctly |
| Recall (sensitivity) | **98.0%** | caught 335 of 342 tumor slices |
| Specificity | **82.2%** | correctly cleared 626 of 762 normal slices |
| Precision | **71.1%** | share of "Tumor" predictions that really were tumors |
| F1 score | 0.824 | balance of precision and recall |
| ROC AUC | 0.904 | how well it ranks tumor slices above normal ones (1.0 is perfect) |
| Loss | 2.078 | categorical cross-entropy on the test set |

Confusion matrix:

|  | Predicted Normal | Predicted Tumor |
|---|---|---|
| **Actual Normal** | 626 | 136 (false alarms) |
| **Actual Tumor** | 7 (missed) | 335 |

The model leans cautious: it rarely misses a tumor, but about 3 in 10 of its tumor flags are false alarms. These numbers come from a 2-epoch run (see [Limitations](#limitations)). Run `dvc repro` to retrain and refresh them, and `dvc metrics show` to print them.

## How it works

```text
Kaggle CT dataset (zip)
        │
        ▼
1. data_ingestion       pick the Normal and Tumor folders, split 70/15/15 into train/val/test
        │
        ▼
2. prepare_base_model   VGG16 (ImageNet weights, no top), freeze it, add Flatten + Dense softmax head
        │
        ▼
3. training             train the head on train/, validate on val/, save model.h5 + classes.json
        │
        ▼
4. evaluation           score test/ and write scores.json (MLflow logging can be switched on)
        │
        ▼
Flask app (app.py)      web page + /predict API that serves artifacts/training/model.h5
```

## Tech stack

| Area | Tools |
|---|---|
| Model | TensorFlow / Keras 2.12, VGG16 transfer learning |
| Pipeline and versioning | DVC |
| Experiment tracking | MLflow 2.2.2 (local `mlruns/` or DagsHub) |
| Metrics | scikit-learn |
| Web app and API | Flask, Flask-Cors, plain HTML/CSS/JavaScript |
| Packaging and deploy | Docker, GitHub Actions, AWS ECR, AWS EC2 (self-hosted runner) |
| Language | Python 3.11 |

## Project structure

```text
.
├── .github/workflows/main.yaml   # CI/CD: train → build Docker image → deploy to EC2
├── .dvc/                         # DVC metadata
├── app.py                        # Flask web app and REST API
├── main.py                       # runs all four pipeline stages in order, without DVC
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC's record of the last pipeline run
├── params.yaml                   # training hyperparameters
├── config/config.yaml            # file paths used by every stage
├── scores.json                   # test-set metrics from the last evaluation
├── Dockerfile, .dockerignore
├── requirements.txt              # dependencies (also installs src/ as a package)
├── setup.py                      # packages src/cnnClassifier
├── template.py                   # scaffolding script that created this folder layout
├── inputImage.jpg                # a sample kidney CT slice
├── data/archive.zip              # the Kaggle dataset (you download it, git-ignored)
├── artifacts/                    # pipeline outputs (git-ignored)
│   ├── data_ingestion/split/{train,val,test}/{Normal,Tumor}/
│   ├── prepare_base_model/       # base_model.h5, base_model_updated.h5
│   └── training/                 # model.h5, classes.json
├── mlruns/                       # local MLflow runs (git-ignored)
├── logs/running_logs.log         # log output of every stage
├── research/                     # notebooks used to build each stage
├── templates/index.html          # the web page
├── static/sample-scan.jpg        # sample scan used by the page
└── src/cnnClassifier/
    ├── __init__.py               # logger: console + logs/running_logs.log
    ├── constants/                # paths to config.yaml and params.yaml
    ├── entity/config_entity.py   # frozen dataclasses holding each stage's settings
    ├── config/configuration.py   # ConfigurationManager: builds those settings from the YAML files
    ├── components/               # the actual work of each stage
    │   ├── data_ingestion.py
    │   ├── prepare_base_model.py
    │   ├── model_training.py
    │   └── model_evaluation_mlflow.py
    ├── pipeline/                 # stage_01 … stage_04 entry points, prediction.py
    └── utils/
        ├── common.py             # YAML/JSON helpers
        └── preprocessing.py      # the one preprocessing function shared by training and serving
```

## Getting started

### Prerequisites

- **Python 3.11.** TensorFlow 2.12 does not install on Python 3.12 or newer.
- About 5 GB of free disk. The dataset zip is 1.6 GB, the split images take about 0.9 GB, the saved models about 0.2 GB, and the Python packages (mostly TensorFlow) need a couple of GB more.
- On Windows, TensorFlow 2.12 runs on the CPU only, which is enough for this project.

### 1. Clone the repository

```bash
git clone https://github.com/theBhavyaaggarwal81/Kidney-Disease-Classification-MLflow-DVC.git
cd Kidney-Disease-Classification-MLflow-DVC
```

### 2. Create a Python 3.11 environment

With conda:

```bash
conda create -n cnncls python=3.11 -y
conda activate cnncls
```

Or with venv on Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Or with venv on macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

The last line of `requirements.txt` (`-e .`) installs `src/cnnClassifier` as an editable package, so the stage scripts can `import cnnClassifier`.

### 4. Get the dataset

Download the Kaggle dataset and save it as `data/archive.zip`. See [Dataset](#dataset) for the details.

### 5. Train the model

```bash
dvc repro
```

This runs the four stages and writes `artifacts/training/model.h5`, `artifacts/training/classes.json` and `scores.json`. `python main.py` runs the same four stages without DVC, so it always reruns everything.

To skip training, download the `trained-model` artifact from a successful GitHub Actions run. Put `model.h5` and `classes.json` in `artifacts/training/` and `scores.json` in the project root.

### 6. Start the web app

```bash
python app.py
```

Then open <http://localhost:8080>. The first prediction takes a few seconds while the model loads.

## Dataset

[CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) on Kaggle, licensed CC BY 4.0. It has four classes. This project uses only the folders listed in `DATA_CLASSES` in `params.yaml`, which are **Normal** and **Tumor**. Cyst and Stone are left out.

Downloading from the website needs a Kaggle login. The Kaggle CLI can download this public dataset without one:

```bash
kaggle datasets download nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone -p data
```

It saves `data/ct-kidney-dataset-normal-cyst-tumor-and-stone.zip`. Rename it to `data/archive.zip`, or point `data_ingestion.source_path` in `config/config.yaml` at the zip or at the folder you extracted it to. Install the Kaggle CLI in a separate environment, because it needs a newer `protobuf` than TensorFlow 2.12 allows. The CI workflow does the same.

How the data is split (`src/cnnClassifier/components/data_ingestion.py`):

- Each class folder is found by name, ignoring case. Each class must come from exactly one folder, so no image can end up in two splits.
- Each class is shuffled with a fixed seed (`SEED: 42`) and split into test (15%), validation (15%) and train (the remaining 70%). Every run produces the same split.
- The split images are copied to `artifacts/data_ingestion/split/`. The test folder is used only by the evaluation stage.

Current split sizes:

| Split | Normal | Tumor | Total |
|---|---|---|---|
| train | 3,553 | 1,599 | 5,152 |
| val | 762 | 342 | 1,104 |
| test | 762 | 342 | 1,104 |
| **all** | 5,077 | 2,283 | 7,360 |

## Training pipeline (DVC)

`dvc.yaml` defines four stages. DVC reruns a stage only when its code, inputs or listed parameters change.

| Stage | Script | Reads | Writes | Parameters |
|---|---|---|---|---|
| `data_ingestion` | `stage_01_data_ingestion.py` | `data/archive.zip` | `artifacts/data_ingestion/split/{train,val,test}` | `DATA_CLASSES`, `VALIDATION_SPLIT`, `TEST_SPLIT`, `SEED` |
| `prepare_base_model` | `stage_02_prepare_base_model.py` | VGG16 ImageNet weights (downloaded by Keras) | `artifacts/prepare_base_model/` | `IMAGE_SIZE`, `INCLUDE_TOP`, `CLASSES`, `WEIGHTS`, `LEARNING_RATE` |
| `training` | `stage_03_model_training.py` | train and val splits, updated base model | `artifacts/training/model.h5`, `classes.json` | `IMAGE_SIZE`, `EPOCHS`, `BATCH_SIZE`, `AUGMENTATION` |
| `evaluation` | `stage_04_model_evaluation.py` | test split, `model.h5` | `scores.json` (DVC metric) | `IMAGE_SIZE`, `BATCH_SIZE`, `POSITIVE_CLASS` |

Useful DVC commands:

| Command | What it does |
|---|---|
| `dvc repro` | runs every stage whose code, inputs or parameters changed |
| `dvc repro -f` | reruns every stage |
| `dvc dag` | draws the stage graph |
| `dvc metrics show` | prints `scores.json` |
| `dvc params diff` | shows parameter changes since the last commit |

Every stage logs to the console and to `logs/running_logs.log`.

## Configuration

### `params.yaml`: hyperparameters

| Key | Value | Meaning |
|---|---|---|
| `AUGMENTATION` | `True` | random rotation, flips, shifts, shear and zoom on training images |
| `IMAGE_SIZE` | `[224, 224, 3]` | network input size (VGG16 expects 224 × 224 RGB) |
| `BATCH_SIZE` | `16` | images per training step |
| `INCLUDE_TOP` | `False` | drop VGG16's ImageNet classifier layers |
| `EPOCHS` | `2` | passes over the training set |
| `CLASSES` | `2` | number of output classes; must equal the length of `DATA_CLASSES` |
| `DATA_CLASSES` | `[Normal, Tumor]` | dataset folders to use |
| `WEIGHTS` | `imagenet` | pretrained weights for VGG16 |
| `LEARNING_RATE` | `0.01` | SGD learning rate |
| `VALIDATION_SPLIT` | `0.15` | share of each class used for validation |
| `TEST_SPLIT` | `0.15` | share of each class held out for evaluation |
| `SEED` | `42` | seed for the split shuffle |
| `POSITIVE_CLASS` | `Tumor` | class treated as positive for recall, specificity, precision and ROC AUC |

`ConfigurationManager` refuses to start if `CLASSES` doesn't match the number of `DATA_CLASSES`.

### `config/config.yaml`: file paths

| Key | Default |
|---|---|
| `artifacts_root` | `artifacts` |
| `data_ingestion.source_path` | `data/archive.zip` (a zip or an extracted folder) |
| `data_ingestion.train_dir` / `val_dir` / `test_dir` | `artifacts/data_ingestion/split/train` / `val` / `test` |
| `prepare_base_model.base_model_path` | `artifacts/prepare_base_model/base_model.h5` |
| `prepare_base_model.updated_base_model_path` | `artifacts/prepare_base_model/base_model_updated.h5` |
| `training.trained_model_path` | `artifacts/training/model.h5` |
| `training.classes_path` | `artifacts/training/classes.json` |

## Model

- **Backbone:** VGG16 with ImageNet weights and without its top classifier. All of its layers are frozen.
- **Head:** `Flatten` (7 × 7 × 512 = 25,088 features) followed by a `Dense` softmax layer with one unit per class. That's 50,178 trainable parameters.
- **Training:** SGD optimizer (learning rate 0.01), categorical cross-entropy loss, batch size 16, 2 epochs.
- **Augmentation** (training images only): rotation up to 40°, horizontal flip, width and height shift 0.2, shear 0.2, zoom 0.2.
- **Preprocessing:** images are resized to 224 × 224 with bilinear interpolation and passed through VGG16's own `preprocess_input` (RGB to BGR plus ImageNet mean subtraction, not a 1/255 rescale). Training, evaluation and prediction all call the same function, `utils/preprocessing.py`, so the model always sees the same kind of input.
- **Class order:** training saves the class names in output order to `artifacts/training/classes.json` (`["Normal", "Tumor"]`), and prediction reads them back from there.

## Experiment tracking (MLflow)

`Evaluation.log_into_mlflow()` in `components/model_evaluation_mlflow.py` logs every parameter in `params.yaml`, every score in `scores.json`, and the Keras model. When the tracking server isn't a local file store, it also registers the model as `VGG16Model`.

**It is switched off by default:** the call is commented out in `src/cnnClassifier/pipeline/stage_04_model_evaluation.py`. To turn it on, uncomment this line:

```python
evaluation.log_into_mlflow()
```

Where runs go:

- **Locally** (no `MLFLOW_TRACKING_URI` set): runs go to `./mlruns`. To browse them, run `mlflow ui` and open <http://127.0.0.1:5000>.
- **On DagsHub:** set these environment variables before running the pipeline.

  macOS / Linux:

  ```bash
  export MLFLOW_TRACKING_URI=https://dagshub.com/theBhavyaaggarwal81/Kidney-Disease-Classification-MLflow-DVC.mlflow
  export MLFLOW_TRACKING_USERNAME=theBhavyaaggarwal81
  export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
  ```

  Windows (PowerShell):

  ```powershell
  $env:MLFLOW_TRACKING_URI = "https://dagshub.com/theBhavyaaggarwal81/Kidney-Disease-Classification-MLflow-DVC.mlflow"
  $env:MLFLOW_TRACKING_USERNAME = "theBhavyaaggarwal81"
  $env:MLFLOW_TRACKING_PASSWORD = "<your-dagshub-token>"
  ```

The model-registry URL is hard-coded as `mlflow_uri` in `src/cnnClassifier/config/configuration.py`. Change it there if you use your own DagsHub repository.

**MLflow vs DVC in this project:** MLflow is production-grade experiment tracking. It records parameters, metrics and models for every run and can register models. DVC is a lightweight way to version data and define the pipeline: it reruns only the stages whose inputs changed.

## Web app

`python app.py` serves the page at <http://localhost:8080>. The page is a single file, [templates/index.html](templates/index.html), with all its CSS and JavaScript inline and no build step.

What's on the page:

- **Intro screen:** a short animated loading screen, shown once per browser session.
- **Hero:** a short summary of the project and a rotating badge around a sample CT scan.
- **Check a Scan:**
  - Drag and drop a JPG or PNG (up to 10 MB), browse for a file, or click **Try the sample scan**.
  - The page sends the image to `/predict` and shows the predicted class, the model's confidence, a bar for each class, and a reminder that the result isn't a diagnosis.
  - Errors are shown in place: a wrong file type, an oversized file, an unreadable image, or the server not running.
- **How It Works:** four cards covering upload, preprocessing, classification and tracking. On wide screens they stack on top of each other as you scroll.
- **Model Report:** accuracy, tumor recall, specificity, ROC AUC and the confusion matrix. They're read from `scores.json` each time the page loads, so they update after you retrain. The section is hidden if `scores.json` is missing.

Other details:

- The layout works on phones, and the animations turn off when the device asks for reduced motion.
- Fonts load from Google Fonts and Fontshare. Without internet, the page falls back to system fonts.
- The sample image the page uses is `static/sample-scan.jpg`, a copy of `inputImage.jpg`.

## REST API

All endpoints are served by `app.py` on port 8080. CORS is enabled.

### `GET /`

Returns the web page.

### `POST /predict`

Classifies one image.

Request body (JSON): `image` holds the raw image file's bytes, base64-encoded (JPG, PNG or any other format Pillow can read).

```json
{ "image": "<base64-encoded image bytes>" }
```

Response `200` (example values):

```json
[
  {
    "image": "Tumor",
    "probabilities": { "Normal": 0.0213, "Tumor": 0.9787 }
  }
]
```

- `image` is the predicted class. The key name is kept from earlier versions so existing clients keep working.
- `probabilities` holds the softmax score for each class. The scores add up to 1.

Errors return `400` with a JSON message:

| Cause | Response |
|---|---|
| no JSON body, or no `image` string | `{"error": "Send JSON with a base64-encoded 'image' field"}` |
| invalid base64, or bytes that aren't an image | `{"error": "Could not read the uploaded image"}` |

Each request is decoded in memory, so nothing is written to disk and concurrent requests can't mix up images. The model loads on the first request and reloads automatically whenever `artifacts/training/model.h5` changes, for example after `/train`.

Example call from Python:

```python
import base64
import requests

with open("inputImage.jpg", "rb") as f:
    payload = {"image": base64.b64encode(f.read()).decode()}

print(requests.post("http://localhost:8080/predict", json=payload).json())
```

### `POST /train`

Starts retraining in the background by running `python main.py`, which runs all four stages. This endpoint is **disabled unless the `TRAIN_TOKEN` environment variable is set**. Callers must send the same value in the `X-Train-Token` header.

```bash
curl -X POST -H "X-Train-Token: $TRAIN_TOKEN" http://localhost:8080/train
```

| Status | Meaning |
|---|---|
| `202` | training started |
| `401` | wrong token |
| `403` | endpoint disabled (`TRAIN_TOKEN` not set) |
| `409` | a training run is already in progress |

Training needs the dataset at `data_ingestion.source_path` (by default `data/archive.zip`) on the machine or container running the app.

## Docker

The image (`python:3.11-slim-bookworm`) only serves predictions, so it needs a trained model. The build fails on purpose if `artifacts/training/model.h5` or `classes.json` is missing.

```bash
dvc repro
docker build -t kidney-classifier .
docker run -p 8080:8080 kidney-classifier
```

`.dockerignore` keeps the image small. It leaves out the dataset, notebooks, MLflow runs, logs and every artifact except `artifacts/training/`. `scores.json` is included, so the page's Model Report works inside the container.

To use `/train` inside a container, set a token and mount the dataset:

```bash
docker run -p 8080:8080 -e TRAIN_TOKEN=<long-random-string> -v "$(pwd)/data:/app/data" kidney-classifier
```

## CI/CD on AWS with GitHub Actions

`.github/workflows/main.yaml` runs on every push to `main`, except pushes that only change `README.md`.

| Job | Runs on | What it does |
|---|---|---|
| Continuous Integration | GitHub runner | placeholder lint and test steps (they only print a message for now) |
| Train Model | GitHub runner | installs Python 3.11 and the requirements, downloads the dataset with the Kaggle CLI (in its own venv), runs `dvc repro`, and uploads `model.h5`, `classes.json` and `scores.json` as the `trained-model` artifact |
| Continuous Delivery | GitHub runner | downloads that artifact, builds the Docker image and pushes it to Amazon ECR as `:latest` |
| Continuous Deployment | your EC2 instance (self-hosted runner) | pulls the image, replaces the running `cnncls` container, starts it on port 8080 with `TRAIN_TOKEN`, and cleans up unused Docker data with `docker system prune` |

### AWS setup

1. **Log in to the AWS console.**
2. **Create an IAM user for deployment** and attach these policies:
   - `AmazonEC2ContainerRegistryFullAccess`, to push and pull images in ECR
   - `AmazonEC2FullAccess`, to run the EC2 virtual machine

   The deployment flow: build the Docker image from the source code, push it to ECR, launch EC2, pull the image from ECR on EC2, and run the container there.
3. **Create an ECR repository** to store the image. Save its URI, for example `<account-id>.dkr.ecr.<region>.amazonaws.com/kidney`.
4. **Create an EC2 machine (Ubuntu).** In its security group, add an inbound rule for TCP port 8080 so the app is reachable.
5. **Install Docker on the EC2 machine:**

   ```bash
   # optional
   sudo apt-get update -y
   sudo apt-get upgrade

   # required
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

6. **Set up EC2 as a self-hosted runner:** in the GitHub repository, go to *Settings → Actions → Runners → New self-hosted runner*, choose Linux, and run the commands it shows on the EC2 machine, one by one.
7. **Add the GitHub secrets** under *Settings → Secrets and variables → Actions*:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | access key of the IAM user |
| `AWS_SECRET_ACCESS_KEY` | secret key of the IAM user |
| `AWS_REGION` | region of your ECR repository, for example `ap-south-1` |
| `AWS_ECR_LOGIN_URI` | registry URI without the repository name, for example `<account-id>.dkr.ecr.<region>.amazonaws.com` |
| `ECR_REPOSITORY_NAME` | name of the ECR repository, for example `kidney` |
| `TRAIN_TOKEN` | optional: any long random string; leave it unset to keep `/train` disabled |
| `KAGGLE_API_TOKEN` | optional: a token from <https://www.kaggle.com/settings/api>; the dataset downloads without it, but anonymous downloads can be rate-limited |

After a deploy, the app runs at `http://<ec2-public-ip>:8080`.

## Development workflow

The order to follow when adding or changing a pipeline stage:

1. Update `config/config.yaml`
2. Update `params.yaml`
3. Update the entity (`src/cnnClassifier/entity/config_entity.py`)
4. Update the configuration manager (`src/cnnClassifier/config/configuration.py`)
5. Update the components (`src/cnnClassifier/components/`)
6. Update the pipeline (`src/cnnClassifier/pipeline/`)
7. Update `main.py`
8. Update `dvc.yaml`
9. Update `app.py`

The notebooks in `research/` show how each stage was first built and tried out.

## Limitations

- **Short training.** The model was trained for only 2 epochs with a frozen backbone. More epochs, or unfreezing the last VGG16 blocks, would likely raise precision.
- **Many false alarms.** About 29% of its "Tumor" predictions on the test set were wrong. The high test loss (2.08) also points to some confident mistakes.
- **Two classes only.** A cyst or stone scan is still forced into Normal or Tumor, because those classes were left out of training.
- **One 2D slice at a time.** The model never sees the full 3D scan or any patient history.
- **Not clinically validated.** The data comes from one public dataset, and the model has not been tested on scans from other hospitals or scanners.
- **Browser-only size limit.** The 10 MB upload limit is checked in the browser only. The API itself accepts any size.

## License and credits

- Dataset: [CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) on Kaggle, CC BY 4.0.
- VGG16: Simonyan and Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* (2014), with ImageNet weights from Keras Applications.
- The repository doesn't include a license file for its code yet.

Thanks!
