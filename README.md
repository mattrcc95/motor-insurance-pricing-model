# Motor insurance pricing model

End-to-end claim **frequency** and **severity** modelling on the public French motor
third-party liability dataset **freMTPL2** (~678,000 policies).

> Work in progress - the README is completed as the notebooks are added.

## Quick start

```powershell
git clone https://github.com/mattrcc95/motor-insurance-pricing-model.git
cd motor-insurance-pricing-model
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ipykernel install --user --name motor-pricing --display-name "Python (motor-pricing)"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\jupyter.exe lab
```
