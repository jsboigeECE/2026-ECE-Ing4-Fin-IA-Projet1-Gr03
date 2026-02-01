# introduction 

Ce projet vise à implémenter une stratégie de market making optimale prenant en compte les contraintes d’inventaire du market makeur. Un Market Makeur propose en permanence des prix d’achat (bid) et de vente (ask), et cherche à maximiser son profit tout en limitant le risque associé à l’inventaire détenu.

L'objectif final est de proposer une stratégie algorithmique capable de :
- Simuler l’évolution du prix d’un actif,
- Calculer des spreads bid/ask optimaux,
- Gérer les risques d’inventaire via une formulation inspirée de l’équation de Hamilton-Jacobi-Bellman (HJB),
- Présenter une solution numérique cohérente et interprétable.

