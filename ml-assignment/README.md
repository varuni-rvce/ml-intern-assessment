# Trigram Model — Quick Start

Prerequisites
- Python 3.8+
- (optional) virtual environment

Install dependencies
```powershell
# from workspace root (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run tests
- Recommended: run tests from the package root so `src` imports resolve automatically:
```powershell
cd ml-assignment
python -m pytest -q
```

- Alternative: run from workspace root by setting PYTHONPATH:
```powershell
$env:PYTHONPATH = "$PWD\ml-assignment"
python -m pytest ml-assignment/tests/test_ngram.py -q
```

Example usage (interactive)
```python
from src.ngram_model import TrigramModel
m = TrigramModel()
m.fit("I am a test sentence. This is another test sentence.")
print(m.generate())   # stochastic output
```

Notes
- The model is a trigram implementation (N=3). Generation is probabilistic and therefore non‑deterministic unless you control the global random seed.
- See evaluation.md for design decisions and testing instructions.