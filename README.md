# mlops-demo

Digit classifier (PyTorch) with experiment tracking (MLflow), tests (pytest),
an API (FastAPI) and CI (GitHub Actions).

## Setup (Windows, VS Code terminal)

    python -m venv .venv
    # activate: Git Bash:   source .venv/Scripts/activate
    #           cmd:        .venv\Scripts\activate.bat
    #           PowerShell: .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run

    python -m src.train          # trains and creates model.pt
    mlflow ui                    # http://127.0.0.1:5000
    pytest
    uvicorn src.serve:app --reload   # http://127.0.0.1:8000/docs
