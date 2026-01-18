# xtream2stremio

Beta v1.0.0

Self-hosted Stremio addon that reads an Xtream Codes portal and serves:
- `/manifest.json`
- `/stream/...`

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp env.example .env
# edit .env with your own values

python xtream2stremio.py

