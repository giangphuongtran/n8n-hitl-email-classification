# GitHub Repository Setup & What to Commit

## 📋 What You Should Commit

### ✅ COMMIT These Files

```
✅ Source Code
   ├── src/email_classification/pipelines/
   │   ├── training/nodes.py
   │   ├── training/pipeline.py
   │   ├── inference/nodes.py
   │   ├── inference/pipeline.py
   │   └── __init__.py
   ├── src/email_classification/pipeline_registry.py
   └── src/email_classification/settings.py

✅ Configuration
   ├── conf/base/catalog.yml
   ├── conf/base/parameters.yml
   └── pyproject.toml

✅ Docker & Deployment
   ├── Dockerfile (updated with Kedro stages)
   ├── docker-compose.yml
   ├── requirements.txt

✅ Scripts & Documentation
   ├── run_app.sh
   ├── run_classifier.sh
   ├── setup-py312.sh
   ├── README.md
   ├── HOW_TO_RUN.md
   ├── RUNNING_AFTER_KEDRO.md
   ├── PYTHON_312_SETUP.md
   ├── KEDRO_QUICKSTART.md
   ├── KEDRO_VISUALIZATION_GUIDE.md
   ├── app.py
   └── tests/

✅ Git Configuration
   └── .gitignore (already has data/ directory ignored)
```

### ❌ DO NOT COMMIT These

```
❌ Virtual Environment
   └── .venv/  ← Already in .gitignore

❌ Data Files
   └── data/  ← Already in .gitignore
      ├── 01_raw/*.csv
      ├── 06_models/
      ├── 07_model_output/
      └── etc.

❌ Cache & Temporary Files
   ├── .cache/
   ├── logs/
   ├── .viz/
   ├── __pycache__/
   ├── *.pyc
   ├── .DS_Store
   └── etc.

❌ Sensitive Data
   ├── conf/local/*credentials*
   ├── .env files
   └── secrets
```

---

## 🔄 Git Workflow

### Initial Setup

```bash
# 1. Navigate to project
cd /Users/mac/Documents/email-classification

# 2. Initialize git (if not already done)
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 3. Add remote repository
git remote add origin https://github.com/yourusername/email-classification.git

# 4. Check current status
git status
```

### First Commit (All Kedro files)

```bash
# 1. Stage all files
git add -A

# 2. Verify what will be committed
git status

# 3. Commit
git commit -m "feat: refactor to Kedro pipelines with visualization

- Implement training pipeline (7 nodes)
- Implement inference pipeline (4 nodes)
- Add Kedro Viz for pipeline visualization
- Update data organization to data/01_raw, data/06_models
- Configure catalog.yml and parameters.yml
- Update Dockerfile with multi-stage builds
- Add Python 3.12 support
- Add comprehensive documentation"

# 4. Push to GitHub
git branch -M main
git push -u origin main
```

### Regular Commits (After Training)

```bash
# After successfully training a model
git add src/ conf/

git commit -m "feat: add training results and model metadata

- Train model with batch_size=16, learning_rate=2e-5
- Accuracy: 0.92 on test set
- Update parameters.yml with new hyperparameters
- Add model metrics to documentation"

# Note: Do NOT commit data/06_models/ (trained model files)
```

---

## 📝 .gitignore Configuration

Your `.gitignore` should already have:

```gitignore
# Kedro
conf/local/**
!conf/local/.gitkeep
.telemetry
.viz
*.log

# Data (all)
data/**
!data/**/
!.gitkeep

# Python
__pycache__/
*.py[cod]
.venv/
.env

# OS
.DS_Store
.AppleDouble

# IDE
.idea/
.vscode/
*.swp
```

This means:
- ✅ `conf/base/` files ARE committed
- ✅ `conf/local/` files are NOT committed
- ❌ `data/` directory files are NOT committed

---

## 🏗️ Recommended GitHub Structure

### Branch Strategy

```
main (production)
  └── All training pipelines
  └── All API code
  └── Documentation
  └── Docker configs

develop (development)
  └── Experimental code
  └── New pipeline nodes
  └── Feature branches
```

### Tags for Model Versions

```bash
# After successful training
git tag -a v1.0.0-model -m "Model trained with 92% accuracy"
git push origin v1.0.0-model

# After API updates
git tag -a v1.0.1-api -m "Add new endpoints for batch classification"
git push origin v1.0.1-api
```

---

## 📄 README for GitHub

Your `README.md` should include:

```markdown
# Email Classification with Kedro

Fine-tune DistilBERT for email classification with Kedro pipeline visualization.

## Quick Start

```bash
# Clone
git clone <your-repo-url>
cd email-classification

# Setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Train
kedro run --pipeline training

# Visualize
kedro viz

# API
python -m uvicorn app:app --port 8001
```

## Documentation

- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Complete workflow guide
- [RUNNING_AFTER_KEDRO.md](RUNNING_AFTER_KEDRO.md) - Docker & examples
- [KEDRO_QUICKSTART.md](KEDRO_QUICKSTART.md) - Quick reference

## Architecture

- Training: Kedro pipeline (7 nodes)
- API: FastAPI server
- Orchestration: N8N workflows
- Visualization: Kedro Viz

## License

[Your License Here]
```

---

## 🚀 Deployment & CI/CD

### GitHub Actions Workflow (Optional)

Create `.github/workflows/train.yml`:

```yaml
name: Train Model

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  train:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt && pip install -e .
      - run: kedro run --pipeline training
      - uses: actions/upload-artifact@v2
        with:
          name: model
          path: data/06_models/
```

---

## 📋 Commit Template

Create `.gitmessage`:

```
[TYPE] Short description (max 50 chars)

Longer explanation of changes (max 72 chars per line)

- Feature 1
- Feature 2

Dataset: [if applicable]
Accuracy: [if applicable]
Hyperparameters: [if applicable]

Fixes #[issue number]
```

Use with:
```bash
git config commit.template .gitmessage
```

---

## 📊 Example Commit History

```
commit abc123  - chore: Update documentation
commit def456  - feat: Add inference pipeline
commit ghi789  - feat: Add Kedro Viz support
commit jkl012  - refactor: Reorganize data directories
commit mno345  - feat: Refactor to Kedro pipelines
commit pqr678  - initial: Email classification API

```

---

## ✅ Pre-Commit Checklist

Before pushing to GitHub:

- [ ] All Kedro pipeline code works locally
- [ ] `git status` shows only intended files
- [ ] No `.env` or credentials files staged
- [ ] No `data/` or `logs/` directories staged
- [ ] No `.venv/` directory staged
- [ ] Tests pass: `pytest tests/`
- [ ] Code is formatted: `python -m black src/`
- [ ] Linting passes: `python -m flake8 src/`

---

## 🔐 Security Notes

### Never Commit:

1. **API Keys/Credentials**
   - Hugging Face tokens
   - Database passwords
   - AWS credentials

2. **Training Data**
   - Customer emails
   - Sensitive information

3. **Model Files**
   - `.safetensors` files (too large)
   - `.pt` checkpoint files

### Instead:

1. Use `.env` files (in `.gitignore`)
2. Store data in S3/GCS (reference in README)
3. Store models in MLflow/Weights & Biases
4. Document setup steps for other developers

---

## 📚 Useful Git Commands

```bash
# Show what will be committed
git status

# Stage specific files
git add src/

# Unstage a file
git reset src/email_classification/old_file.py

# View diff before committing
git diff --staged

# View commit history
git log --oneline

# Amend last commit (if not pushed)
git commit --amend

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View file at specific commit
git show abc123:src/file.py

# Create and push tag
git tag -a v1.0.0 -m "Version 1.0.0"
git push origin v1.0.0
```

---

## 📞 Next Steps

1. **Initialize Git:**
   ```bash
   cd /Users/mac/Documents/email-classification
   git init
   git add -A
   git commit -m "Initial commit: Kedro email classification pipeline"
   ```

2. **Add Remote:**
   ```bash
   git remote add origin https://github.com/yourusername/email-classification
   git push -u origin main
   ```

3. **Protect Main Branch** (in GitHub Settings):
   - Require pull request reviews
   - Require status checks to pass
   - Dismiss stale review approvals

4. **Set Up CI/CD** (optional):
   - Add GitHub Actions workflows
   - Enable automated testing
   - Setup artifact uploads

---

**All set! Your Kedro pipeline is ready for GitHub!** 🚀
