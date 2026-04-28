# After Kedro Refactoring - Complete Workflow Guide

## 🎯 Quick Summary

| Task | Old Way | New Way |
|------|---------|---------|
| Train model | `python classifier.py ...` | `source .venv/bin/activate && kedro run --pipeline training` |
| Run API | `python app.py` | `source .venv/bin/activate && python -m uvicorn app:app --port 8001` |
| Visualize | N/A | `source .venv/bin/activate && kedro viz` |

---

## ✅ How to Run Everything

### 1️⃣ Setup (First Time Only)

```bash
# Navigate to project
cd /Users/mac/Documents/email-classification

# Activate virtual environment
source .venv/bin/activate

# Verify Python 3.12
python --version  # Should show Python 3.12.x

# Verify Kedro
kedro --version
```

### 2️⃣ Train Model (Replaces classifier.py)

**Recommended approach:**
```bash
# 1. Make sure venv is activated
source .venv/bin/activate

# 2. Ensure training data exists at:
#    data/01_raw/n8n-classifier-tracker.csv

# 3. Run Kedro training pipeline
kedro run --pipeline training
```

**Or use the wrapper script (with source):**
```bash
source .venv/bin/activate
source run_classifier.sh
```

**What it does:**
- Loads training data from `data/01_raw/`
- Tokenizes with DistilBERT
- Fine-tunes model on your data
- Saves model to `data/06_models/email_classifier_finetuned/`
- Saves metrics to `data/07_model_output/evaluation_metrics.json`

**Monitor progress:**
```bash
# In another terminal, watch the pipeline
kedro viz
# Open http://localhost:4141 in browser
```

### 3️⃣ Run API Server (app.py - unchanged)

**Recommended approach:**
```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Start FastAPI server
python -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

**Or use the wrapper script (with source):**
```bash
source .venv/bin/activate
source run_app.sh 8001
```

**Access:**
- 📚 Swagger Docs: http://localhost:8001/docs
- 📖 ReDoc: http://localhost:8001/redoc
- 🔍 Health Check: http://localhost:8001/health

### 4️⃣ View Pipeline Visualization

```bash
source .venv/bin/activate
kedro viz
```

Opens at: **http://localhost:4141**

See all 7 training steps, data flow, and node dependencies.

---

## 📂 Data Organization

Your data flows through these directories:

```
data/
├── 01_raw/
│   └── n8n-classifier-tracker.csv  ← PUT YOUR TRAINING DATA HERE
├── 02_intermediate/
│   └── (intermediate processing files)
├── 03_primary/
│   └── (train/test splits)
├── 05_model_input/
│   └── (tokenized datasets)
├── 06_models/
│   └── email_classifier_finetuned/  ← MODEL OUTPUT
│       ├── config.json
│       ├── model.safetensors
│       ├── label_mapping.json
│       └── tokenizer.json
├── 07_model_output/
│   └── evaluation_metrics.json      ← METRICS
└── 08_reporting/
    └── (reports and visualizations)
```

**Important:** All paths in `conf/base/parameters.yml` use `data/01_raw/`, `data/06_models/`, etc.

---

## 🐳 Docker

### Run Training in Docker

```bash
# Build training image
docker build -t email-classifier:training --target training .

# Run training (keeps model in ./data/06_models/)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.cache:/app/.cache \
  email-classifier:training
```

### Run API in Docker

```bash
# Build API image
docker build -t email-classifier:api --target api .

# Run API
docker run --rm \
  -p 8001:8001 \
  -v $(pwd)/data/06_models:/app/data/06_models \
  email-classifier:api
```

### Docker Compose (N8N + API)

```bash
# Start N8N (workflow) + API
docker-compose up n8n email-ai-api

# Access N8N: http://localhost:5678
# Access API: http://localhost:8001/docs

# Train only
docker-compose run --rm trainer
```

---

## 🔧 Configuration

Edit `conf/base/parameters.yml`:

```yaml
# Data paths (relative to project root)
csv_path: "data/01_raw/n8n-classifier-tracker.csv"
output_dir: "data/06_models/email_classifier_finetuned"

# Training hyperparameters
num_epochs: 3
batch_size: 16
learning_rate: 0.00002
max_length: 512

# Test/train split
test_size: 0.2

# Model name
model_name: "distilbert-base-multilingual-cased"
```

### Change parameters for different experiments:

```bash
# Edit the file
nano conf/base/parameters.yml

# Change num_epochs from 3 to 5, then run:
kedro run --pipeline training
```

---

## 📚 Kedro Pipeline Workflow

### Run All Steps
```bash
kedro run --pipeline training
```

### Run Only Specific Steps
```bash
# Data preparation (load + split)
kedro run --pipeline training --tags data_preparation

# Model setup + tokenization
kedro run --pipeline training --tags model_setup

# Training only
kedro run --pipeline training --tags training

# Evaluation only
kedro run --pipeline training --tags evaluation
```

### Run Inference Pipeline
```bash
# Classify new emails
kedro run --pipeline inference
```

---

## 🚀 Example Workflows

### Workflow 1: First Time Training

```bash
# 1. Navigate to project
cd /Users/mac/Documents/email-classification

# 2. Copy your training CSV to the right location
cp ~/Downloads/emails_with_labels.csv data/01_raw/n8n-classifier-tracker.csv

# 3. Activate venv
source .venv/bin/activate

# 4. Run training
kedro run --pipeline training

# 5. Watch progress
# In another terminal:
kedro viz
# Open http://localhost:4141

# 6. Check outputs
ls -la data/06_models/email_classifier_finetuned/
cat data/07_model_output/evaluation_metrics.json
```

### Workflow 2: Iterative Model Tuning

```bash
source .venv/bin/activate

# 1. Start Kedro Viz in background
kedro viz &

# 2. Try with different parameters
nano conf/base/parameters.yml  # Change batch_size, learning_rate, etc.

# 3. Run pipeline
kedro run --pipeline training

# 4. Check metrics
cat data/07_model_output/evaluation_metrics.json

# 5. Adjust and repeat
nano conf/base/parameters.yml
kedro run --pipeline training
```

### Workflow 3: Train and Deploy

```bash
# Terminal 1: Train
source .venv/bin/activate
cd data/01_raw/ && download-my-latest-data.sh
cd ../..
kedro run --pipeline training

# Terminal 2: Start API with trained model
source .venv/bin/activate
python -m uvicorn app:app --port 8001

# Terminal 3: Test API
curl http://localhost:8001/docs
```

---

## ✅ What Changed vs. What Didn't

### ✅ Completely Changed
- **Training**: Now uses Kedro pipeline instead of `python classifier.py`
- **Data organization**: Must follow `data/01_raw/`, `data/06_models/` structure
- **Configuration**: Centralized in `conf/base/parameters.yml` instead of CLI args
- **Visualization**: Can now see pipeline DAG with `kedro viz`

### ✅ Still Works Exactly the Same
- **API (`app.py`)**: Unchanged, still works as FastAPI server
- **Docker**: Updated but still builds images and runs containers
- **N8N Integration**: Still works with the API

---

## 📝 Cheat Sheet

```bash
# Activate environment
source .venv/bin/activate

# Train model
kedro run --pipeline training

# Run API
python -m uvicorn app:app --port 8001

# View pipeline
kedro viz

# Run specific steps only
kedro run --pipeline training --tags training

# Check data location
ls -la data/06_models/email_classifier_finetuned/

# Docker train
docker build -t img --target training . && docker run --rm -v $(pwd)/data:/app/data img

# Docker API
docker build -t img --target api . && docker run -p 8001:8001 -v $(pwd)/data/06_models:/app/data/06_models img
```

---

## ⚠️ Important Notes

1. **Always activate venv first:** `source .venv/bin/activate`
2. **Always use Python 3.12:** Check with `python --version`
3. **Data path must be correct:** Use `data/01_raw/n8n-classifier-tracker.csv`
4. **Parameters are in YAML:** Edit `conf/base/parameters.yml` not CLI args
5. **Models are in `data/06_models/`:** API will look there for trained models

---

## 🆘 Troubleshooting

### "command not found: python"
```bash
# Solution: Activate venv first
source .venv/bin/activate
```

### "No such file: data/01_raw/n8n-classifier-tracker.csv"
```bash
# Solution: Copy your CSV to the right location
cp your_file.csv data/01_raw/n8n-classifier-tracker.csv
```

### "Kedro command not found"
```bash
# Solution: Install in venv
source .venv/bin/activate
pip install kedro kedro-viz
```

### "Port 4141 already in use"
```bash
# Solution: Use different port
kedro viz --port 4142
```

### "ModuleNotFoundError: email_classification"
```bash
# Solution: Install project in editable mode
source .venv/bin/activate
pip install -e .
```

---

## 📞 Next Steps

1. ✅ Place training data in `data/01_raw/n8n-classifier-tracker.csv`
2. ✅ Run `source .venv/bin/activate && kedro run --pipeline training`
3. ✅ Monitor with `kedro viz` in another terminal
4. ✅ Start API with `python -m uvicorn app:app --port 8001`
5. ✅ Test at http://localhost:8001/docs

Enjoy your Kedro-powered ML pipeline! 🚀
