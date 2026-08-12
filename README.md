# BANK Credit Risk & Cross-Sell Engine

## Run backend
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Run frontend
Open `frontend/index.html` with a local static server. For example:
```bash
python -m http.server 5500 --directory frontend
```
Then open http://127.0.0.1:5500

Backend API: http://127.0.0.1:8000

Demo:
- Customer: any username/password
- Staff: staff / staff123
- Developer: developer / dev123

The supplied model was serialized with scikit-learn 1.6.1, so that version is pinned.
