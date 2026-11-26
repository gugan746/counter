# Deploy on Render (no Docker)

Render settings:
- Language: Python 3
- Build Command: pip install -r requirements.txt
- Start Command: gunicorn app:app --workers 1 --threads 2 --timeout 180
- Instance type: Free (or $7 if it runs out of memory)
- Environment variables:
    APP_PASSWORD = (your password; login username is "admin")
    PYTHON_VERSION = 3.12.10
