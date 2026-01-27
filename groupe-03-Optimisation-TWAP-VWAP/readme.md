# Sujet : 50. Optimisation d'exécution d'ordres par contraintes (TWAP/VWAP)
# Membres :
Ilhan Taskin  Farhan Morisson
ING4 Groupe 3

Description du problème et contexte L'exécution optimale de gros ordres boursiers nécessite de découper les transactions en sous-ordres pour minimiser l'impact sur le marché. Les stratégies TWAP (Time-Weighted Average Price) et VWAP (Volume-Weighted Average Price) se formulent comme des problèmes d'optimisation sous contraintes (volume, timing, coût d'impact) où la programmation par contraintes permet d'intégrer des règles de trading complexes.

Références multiples

Stanford : Volume Weighted Average Price Optimal Execution - Boyd et al.
IJCAI : An End-to-End Optimal Trade Execution Framework - IJCAI 2020
Columbia : An Optimal Control Strategy for Execution of Large Stock Orders Using LSTMs - Columbia Finance
Safe Execution : Safe and Compliant Cross-Market Trade Execution via Constrained RL - arXiv 2025
Approches suggérées

Modéliser le problème d'exécution comme CSP avec contraintes de volume et timing
Implémenter les stratégies TWAP et VWAP comme baselines
Développer une optimisation sous contraintes d'impact de marché
Comparer avec des approches de contrôle optimal et reinforcement learning
Technologies pertinentes

Python avec OR-Tools ou cvxpy pour l'optimisation sous contraintes
Backtrader ou vectorbt pour le backtesting
Données tick-by-tick (Binance, Alpaca) pour validation
Pandas pour l'analyse de séries temporelles financières