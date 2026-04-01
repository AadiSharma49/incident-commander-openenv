---
title: Incident Commander OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Incident Commander OpenEnv

`incident-commander-openenv` is a production-style OpenEnv environment for incident response simulation. It exposes a deterministic FastAPI environment, typed Pydantic models, task-specific grading, and a root-level inference script suitable for OpenEnv submission and GitHub sharing.

## Overview

This project simulates real operational incidents where an agent must:

1. inspect evidence
2. diagnose the issue
3. apply mitigation
4. validate recovery
5. communicate status

All task transitions are deterministic. Rewards are incremental. Grading is reproducible.

## Tasks

The environment includes three tasks:

1. `tls-certificate-expiry`  
   Easy task focused on certificate renewal and HTTPS recovery.
2. `db-pool-exhaustion`  
   Medium task focused on deployment rollback and database pool pressure.
3. `idp-outage-vs-security`  
   Hard task focused on distinguishing vendor outage from actual compromise.

## Observation Space

The environment returns a typed `Observation` model with:

- `task_id`
- `difficulty`
- `system_status`
- `alerts`
- `metrics`
- `available_actions`
- `known_facts`
- `completed_actions`
- `turns_remaining`

## Action Space

The environment accepts a typed `Action` model with:

- `action_type`
- `target`
- `note`

Supported `action_type` values:

- `inspect`
- `diagnose`
- `mitigate`
- `validate`
- `communicate`

The environment also supports action normalization for common synonyms such as:

- `tls`, `ssl`, `cert`
- `db`, `database`
- `idp`, `vendor`

## Reward Model

The environment returns a typed `Reward` model with:

- `score`
- `reason`

Reward policy:

- correct diagnosis: positive reward
- correct mitigation: positive reward
- successful validation and recovery: high reward
- repeated useless action: penalty
- invalid or incorrect action: penalty

Task graders return a deterministic score from `0.0` to `1.0`.

## API Endpoints

The FastAPI server exposes:

- `POST /reset`
- `POST /step`
- `GET /state`

## Project Structure

```text
.
├── app
│   ├── env
│   │   ├── __init__.py
│   │   └── env.py
│   ├── graders
│   │   ├── __init__.py
│   │   └── graders.py
│   ├── tasks
│   │   ├── __init__.py
│   │   └── tasks.py
│   ├── __init__.py
│   └── models.py
├── Dockerfile
├── inference.py
├── main.py
├── openenv.yaml
├── README.md
└── requirements.txt
```

## Local Setup

### Windows PowerShell

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Start the API server

```powershell
uvicorn main:app --host 127.0.0.1 --port 7860
```

### Run the inference script

```powershell
python inference.py
```

Expected output pattern:

```text
tls-certificate-expiry: score=1.00 reward_total=... done=True plan=baseline-plan
db-pool-exhaustion: score=1.00 reward_total=... done=True plan=baseline-plan
idp-outage-vs-security: score=1.00 reward_total=... done=True plan=baseline-plan
average_score: 1.00
```

## API Testing

### Reset a task

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:7860/reset" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"task_id":"tls-certificate-expiry"}'
```

### Take one step

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:7860/step" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"action_type":"inspect","target":"tls"}'
```

### Inspect current state

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:7860/state" -Method Get
```

## Docker

Build and run:

```powershell
docker build -t incident-commander-openenv .
docker run --rm -p 7860:7860 incident-commander-openenv
```

If Docker fails with a pipe or engine error, start Docker Desktop first and retry.

## Inference Configuration

`inference.py` supports optional OpenAI-compatible configuration through:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example:

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4o-mini"
$env:HF_TOKEN = "your_token"
python inference.py
```

If these variables are not set, the deterministic baseline still runs.

## GitHub Hygiene

The repo is configured to avoid committing local-only files such as:

- virtual environments
- Python cache files
- editor settings
- logs
- local secrets files

Do not commit:

- `.venv/`
- `__pycache__/`
- `.env`
- `.env.*`
- local IDE folders
- runtime logs

## Submission Files

The important project files are:

- `app/env/env.py`
- `app/tasks/tasks.py`
- `app/graders/graders.py`
- `app/models.py`
- `main.py`
- `inference.py`
- `openenv.yaml`
- `Dockerfile`
- `README.md`
- `requirements.txt`
