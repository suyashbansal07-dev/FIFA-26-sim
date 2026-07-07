# Contributing

## Local checks

```powershell
py -3.13 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python test_wc_sim.py
.venv\Scripts\python -c "import pathlib, py_compile; [py_compile.compile(str(p), doraise=True) for p in pathlib.Path('.').glob('*.py')]"
```

## Ground rules

- Keep model changes evidence-first: add or update a focused self-check.
- Do not commit generated `output/` artifacts or local `.env` files.
- Keep public docs honest about leakage, validation, and limitations.
- Prefer small, reviewable changes over broad rewrites.
