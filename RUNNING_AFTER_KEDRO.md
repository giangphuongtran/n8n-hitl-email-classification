# Running After Kedro Refactoring - Complete Guide

## 🎯 Overview

After Kedro refactoring:
- **`classifier.py`** → `kedro run --pipeline training`
- **`app.py`** → `python -m uvicorn app:app` (unchanged, still works)
- **Docker** → Updated to support both training and API modes
- **Data** → Organized in `data/01_raw/`, `data/06_models/`, etc.

---

## 📚 How to Run Everything

### 1️⃣ Training the Model (replaces classifier.py)

#### Local (with Python 3.12 venv)
```bash
# Activate venv
source .venv/bin/activate

# Verify training data exists
# Place your CSV in: data/01_raw/n8n-classifier-tracker.csv

# Run training pipeline via Kedro
kedro run --pipeline training

# Or use the wrapper script
bash run_classifier.sh
```

**Where does it save?**
- Model: `data/06_models/email_classifier_finetuned/`
- Metrics: `data/07_model_output/evaluation_metrics.json`
- Logs: `logs/`

#### Docker
```bash
# Build training image
docker build -t email-classifier:training --target training .

# Run training (mount volume to preserve model)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.cache:/app/.cache \
  email-classifier:training

# Model saved to: ./data/06_models/email_classifier_finetuned/
```

#### Docker Compose
```bash
# Option in updated docker-compose.yml
docker-compose run --rm trainer
```

---

### 2️⃣ Running the API (app.py)

#### Local (unchanged)
```bash
# Activate venv
source .venv/bin/activate

# Run FastAPI server
python -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload

# Or use wrapper script
bash run_app.sh 8001
```

**Access:**
- API Docs: http://localhost:8001/docs
- API: http://localhost:8001/

#### Docker
```bash
# Build API image
docker build -t email-classifier:api --target api .

# Run API container
docker run --rm \
  -p 8001:8001 \
  -v $(pwd)/data/06_models:/app/data/06_models \
  email-classifier:api
```

#### Docker Compose
```bash
# Run API + N8N
docker-compose up email-ai n8n

# Access:
# - API: http://localhost:8001/docs
# - N8N: http://localhost:5678/
```

---

### 3️⃣ Visualizing Pipelines (Kedro Viz)

```bash
# Activate venv
source .venv/bin/activate

# Start interactive dashboard
kedro viz

# Opens at: http://localhost:4141
```

You'll see:
- Training pipeline DAG
- All 7 nodes and their dependencies
- Data flow between nodes

---

### 4️⃣ Running Specific Pipeline Steps

```bash
# Data preparation only
kedro run --pipeline training --tags data_preparation

# Only model training
kedro run --pipeline training --tags training

# Only evaluation
kedro run --pipeline training --tags evaluation

# Inference pipeline (classify new emails)
kedro run --pipeline inference
```

---

## 🐳 Docker Setup

### Build Images

```bash
# API image (for running the FastAPI server)
docker build -t email-classifier:api --target api .

# Training image (for running Kedro training)
docker build -t email-classifier:training --target training .

# Full build (all stages)
docker build -t email-classifier .
```

### Docker Compose Services

Updated `docker-compose.yml` includes:
- **n8n**: Workflow automation (port 5678)
- **email-ai-api**: FastAPI server (port 8001)
- **email-ai-trainer**: Kedro training (runs once)

```bash
# Start everything
docker-compose up -d

# Run only training
docker-compose run --rm trainer

# Run only API
docker-compose up email-ai-api

# Run N8N + API
docker-compose up n8n email-ai-api
```

---

## 📂 Data Organization

```
data/
├── 01_raw/
│   └── n8n-classifier-tracker.csv  ← YOUR TRAINING DATA HERE
├── 02_intermediate/
│   └── prepared_data.csv
├── 03_primary/
│   ├── train_dataset.pkl
│   ├── test_dataset.pkl
│   └── label_encoder.pkl
├── 05_model_input/
│   ├── train_tokenized.pkl
│   └── test_tokenized.pkl
├── 06_models/
│   └── email_classifier_finetuned/  ← MODEL OUTPUT
│       ├── config.json
│       ├── model.safetensors
│       ├── label_mapping.json
│       └── tokenizer.json
├── 07_model_output/
│   └── evaluation_metrics.json      ← METRICS
└── 08_reporting/
    └── model_performance_report.txt
```

---

## ⚙️ Configuration

Edit `conf/base/parameters.yml`:

```yaml
# Data location
csv_path: "data/01_raw/n8n-classifier-tracker.csv"

# Model output
output_dir: "data/06_models/email_classifier_finetuned"

# Training params
num_epochs: 3
batch_size: 16
learning_rate: 0.00002
```

---

## 🚀 Workflow Examples

### Example 1: Train Model Locally
```bash
# 1. Place data
cp ~/Desktop/emails.csv data/01_raw/n8n-classifier-tracker.csv

# 2. Activate environment
source .venv/bin/activate

# 3. Train
bash run_classifier.sh

# 4. Model saved to data/06_models/email_classifier_finetuned/
```

### Example 2: Train in Docker, Use in API
```bash
# 1. Train in Docker
docker-compose run --rm trainer

# 2. Model saved to ./data/06_models/

# 3. Start API using trained model
docker-compose up email-ai-api

# 4. Send classification requests
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"subject": "Save 50%!", "body": "Limited time offer..."}'
```

### Example 3: Iterative Development
```bash
# 1. Start Kedro Viz
kedro viz &

# 2. Monitor in browser: http://localhost:4141

# 3. Run specific nodes
kedro run --pipeline training --tags data_preparation

# 4. Check outputs in data/

# 5. Adjust parameters in conf/base/parameters.yml

# 6. Run again
kedro run --pipeline training
```

---

## ✅ Quick Checklist

- [ ] Training data in `data/01_raw/n8n-classifier-tracker.csv`
- [ ] Virtual environment activated: `source .venv/bin/activate`
- [ ] Python 3.12: `python --version`
- [ ] Kedro installed: `kedro --version`
- [ ] Run training: `kedro run --pipeline training`
- [ ] Model saved to: `data/06_models/`
- [ ] Start API: `python -m uvicorn app:app`
- [ ] Test API: http://localhost:8001/docs

---

## 📝 Summary

| Task | Command | Old Way |
|------|---------|---------|
| Train | `kedro run --pipeline training` | `python classifier.py ...` |
| API | `python -m uvicorn app:app` | `python app.py` |
| Viz | `kedro viz` | N/A |
| Docker Train | `docker build -t img --target training` | `docker run ... classifier.py` |
| Docker API | `docker build -t img --target api` | `docker build -t img ...` |

---

**Everything is backward compatible!** The API (`app.py`) works exactly the same. Only the training workflow changed to use Kedro.
