# Projet #40 — Optimisation de portefeuille avec contraintes réelles (CSP/MILP)

## Objectif
Comparer une approche convexe (Markowitz/CVXPY) avec une approche CSP/MILP (OR-Tools CP-SAT) intégrant des contraintes réalistes :
- cardinalité (max K actifs)
- lots discrets (q_i entiers)
- contraintes sectorielles
- coûts de transaction (turnover)
- contrôle du risque via CVaR (sur scénarios)
- backtest out-of-sample walk-forward

## Run
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py
python main_compare.py
python main_ml.py
