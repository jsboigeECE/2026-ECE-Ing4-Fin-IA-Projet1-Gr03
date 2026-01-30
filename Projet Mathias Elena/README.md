# 💰 Wealth Planner AI : Optimisation d'Investissement Multi-Périodes

Projet d'Expertise en Développement Quantitatif et Recherche Opérationnelle (ECE 2026).

Ce projet propose une plateforme avancée de simulation et d'optimisation de gestion de patrimoine. Il permet de concevoir des stratégies d'allocation d'actifs optimales sur le long terme, en tenant compte des objectifs de vie, des contraintes de liquidité et des frais de transaction.

---

## 🌟 Fonctionnalités Clés

- **Multi-Actifs** : Support de 6 classes d'actifs (Actions, Obligations, Cash, Or, Crypto, SCPI).
- **Plan de Vie Dynamique** : Intégration d'événements de cash-flow personnalisables (achat immobilier, études, retraite).
- **Moteurs d'Optimisation Avancés** : Comparaison entre Programmation Dynamique, Optimisation Linéaire et Reinforcement Learning.
- **Interface Interactive** : Dashboard Streamlit pour une configuration sans code.
- **Analyse de Risque** : Stress tests, calcul du Sharpe Ratio et visualisations de convergence.

---

## 📂 Architecture du Projet

Le projet est structuré de manière modulaire pour séparer la logique métier des algorithmes de résolution :

```text
Projet Mathias Elena/
├── src/
│   ├── core/
│   │   ├── config.py       # Paramètres par défaut (marché, actifs, frais)
│   │   └── model.py        # Modélisation mathématique du MDP (Transitions, Utilité)
│   ├── solvers/
│   │   ├── base.py         # Interface abstraite pour les solveurs
│   │   ├── dp_solver.py    # Programmation Dynamique (Induction arrière de Bellman)
│   │   ├── ortools_solver.py # Optimisation Linéaire via Google OR-Tools
│   │   └── rl_solver.py    # Reinforcement Learning (PPO via Stable-Baselines3)
│   ├── simulation/
│   │   └── engine.py       # Moteur de simulation Monte Carlo
│   ├── utils/
│   │   └── plotting.py     # Fonctions de visualisation (Matplotlib & Seaborn)
│   └── analysis/
│       ├── stress_analysis.py # Scénarios de crise et stress tests
│       └── plot_robustness.py # Comparaison de la robustesse des solveurs
├── dashboard.py            # Interface Web interactive (Streamlit)
├── main.py                 # Point d'entrée CLI pour le benchmark complet
├── requirements.txt        # Dépendances Python
└── REPORT.md               # Rapport d'analyse détaillé
```

---

## 🛠️ Installation Détaillée

### 1. Prérequis
- Python 3.10 ou supérieur installé.
- Un terminal (PowerShell recommandé sur Windows).

### 2. Clonage et Configuration
```powershell
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement
# Sur Windows (PowerShell) :
.\venv\Scripts\Activate.ps1
# Sur macOS/Linux :
source venv/bin/activate

# Installation des dépendances
pip install -r "Projet Mathias Elena/requirements.txt"
```

---

## 🚀 Guide d'Utilisation

### Option A : Interface Web Interactive (Recommandé)
C'est la méthode la plus simple pour tester vos propres scénarios.
```powershell
cd "Projet Mathias Elena"
..\venv\Scripts\streamlit run dashboard.py
```
**Dans l'interface :**
1. Ajustez votre **Capital Initial** et votre **Épargne Mensuelle** dans la barre latérale.
2. Définissez votre **Âge** et votre **Horizon de Retraite**.
3. Modifiez le tableau des **Événements de Vie** (ajoutez des lignes pour vos projets).
4. Cliquez sur **"Calculer la Stratégie Optimale"**.
5. Explorez les onglets **Richesse**, **Allocation** et **Comparaison**.

### Option B : Benchmark complet via CLI
Pour générer tous les rapports et fichiers CSV de comparaison :
```powershell
python "Projet Mathias Elena/main.py"
```
Les résultats seront générés dans le dossier `Projet Mathias Elena/output/`.

---

## 🧠 Les Moteurs d'Intelligence Artificielle

### 1. Programmation Dynamique (DP)
Utilise l'**Équation de Bellman** pour résoudre le problème par induction arrière. Elle discrétise l'espace de richesse et calcule, pour chaque état possible, l'action qui maximise l'utilité future espérée.
- **Force** : Garantie d'optimalité théorique.
- **Usage** : Idéal pour les horizons longs et les sorties de cash prévisibles.

### 2. Optimisation Linéaire (OR-Tools)
Résout un problème d'optimisation moyenne-variance à chaque pas de temps.
- **Force** : Vitesse d'exécution instantanée.
- **Usage** : Très efficace pour des rééquilibrages fréquents sous contraintes strictes.

### 3. Reinforcement Learning (RL)
Un agent **PPO (Proximal Policy Optimization)** apprend par essai-erreur dans un environnement simulé (Gymnasium).
- **Force** : Capacité à découvrir des stratégies complexes et robustes face à la volatilité.
- **Usage** : Meilleure gestion du risque (Sharpe Ratio élevé).

---

## 📊 Interprétation des Graphiques

- **Stacked Area Chart** : Montre comment votre portefeuille doit évoluer. Par exemple, une réduction des actions à l'approche d'un achat immobilier ou de la retraite.
- **Convergence de Richesse** : La ligne pleine est la moyenne, la zone d'ombre représente l'incertitude (risque). Plus la zone est étroite, plus la stratégie est sûre.
- **Violin Plot** : Compare la dispersion de la richesse finale. Un "violon" haut et fin indique une performance élevée avec peu de risque de mauvaise surprise.

---

## 📝 Notes Techniques
- **Frais de transaction** : Le modèle intègre des frais d'achat/vente (ex: 10% pour les SCPI) pour éviter les mouvements inutiles.
- **Inflation** : Les calculs tiennent compte d'un taux d'inflation pour refléter le pouvoir d'achat réel.
- **Liquidité** : Une pénalité est appliquée en cas de vente forcée d'actifs illiquides (SCPI) pour couvrir un besoin de cash immédiat.

---

## 👥 Contributeurs
Projet réalisé par le Groupe 3 - ECE Ing4 Finance & IA.
Expertise en Développement Quantitatif et Recherche Opérationnelle.
