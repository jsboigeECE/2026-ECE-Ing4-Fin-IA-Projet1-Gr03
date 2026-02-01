# Guide d'Utilisation RL avec Données LOB Réelles

Ce guide explique comment utiliser la partie Reinforcement Learning (RL) du projet avec des données réelles de Limit Order Book (LOB) provenant de LOBSTER ou Binance.

## Table des Matières

1. [Introduction](#introduction)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Sources de Données LOB](#sources-de-données-lob)
5. [Chargement des Données](#chargement-des-données)
6. [Environnement RL](#environnement-rl)
7. [Entraînement d'un Agent RL](#entraînement-dun-agent-rl)
8. [Évaluation](#évaluation)
9. [Exemples Pratiques](#exemples-pratiques)
10. [Dépannage](#dépannage)

---

## Introduction

Le projet fournit maintenant une implémentation complète pour l'entraînement d'agents de market making par reinforcement learning utilisant des données LOB réelles. Les composants principaux sont :

- **`LOBDataLoader`** : Module de chargement de données LOB (LOBSTER/Binance)
- **`LOBMarketMakingEnv`** : Environnement Gymnasium compatible avec les données réelles
- **`train_rl_lob.py`** : Script d'entraînement complet avec monitoring

---

## Prérequis

### Python et Dépendances

- Python 3.11 ou supérieur
- NumPy, SciPy, Pandas
- Matplotlib
- Gymnasium
- stable-baselines3 (pour l'entraînement RL)

### Données LOB

Vous aurez besoin de données LOB réelles :
- **LOBSTER** : Données historiques de carnet d'ordres
- **Binance** : Données de profondeur du carnet d'ordres

---

## Installation

### 1. Installer les Dépendances de Base

```bash
cd market-making-inventory
pip install -r requirements.txt
```

### 2. Installer les Dépendances RL

```bash
pip install stable-baselines3 gymnasium shimmy
```

### 3. Vérifier l'Installation

```bash
python -c "from src.data.lob_loader import LOBDataLoader; print('LOB loader OK')"
python -c "from src.rl_env.lob_market_making_env import LOBMarketMakingEnv; print('RL env OK')"
```

---

## Sources de Données LOB

### LOBSTER

LOBSTER fournit des données historiques de carnet d'ordres pour les actions américaines.

**Format des fichiers :**
- `message_file` : Contient les événements de marché (soumissions, annulations, exécutions)
- `orderbook_file` : Contient les snapshots du carnet d'ordres à chaque événement

**Téléchargement :**
1. Visitez [LOBSTER Data](https://lobsterdata.com/)
2. Sélectionnez l'action et la date
3. Téléchargez les fichiers message et orderbook

**Structure des fichiers :**

Message file (format espace) :
```
time type order_id size price direction
```

Orderbook file (format espace) :
```
ask_price_1 ask_size_1 ... ask_price_10 ask_size_10 bid_price_1 bid_size_1 ... bid_price_10 bid_size_10
```

### Binance

Binance fournit des données de profondeur du carnet d'ordres via leur API.

**Téléchargement via API :**

```python
import requests
import json
from datetime import datetime

# Télécharger les snapshots de profondeur
symbol = "BTCUSDT"
limit = 1000

url = f"https://api.binance.com/api/v3/depth"
params = {"symbol": symbol, "limit": limit}

response = requests.get(url, params=params)
data = response.json()

# Sauvegarder
with open(f"binance_{symbol}_{datetime.now().strftime('%Y%m%d')}.json", "w") as f:
    json.dump(data, f)
```

**Format JSON :**
```json
{
  "lastUpdateId": 123456,
  "bids": [["price1", "qty1"], ["price2", "qty2"], ...],
  "asks": [["price1", "qty1"], ["price2", "qty2"], ...]
}
```

---

## Chargement des Données

### Chargement de Données LOBSTER

```python
from src.data.lob_loader import create_lobster_loader

# Créer le loader
loader = create_lobster_loader(
    message_file='data/AAPL_2012-06-21_message.csv',
    orderbook_file='data/AAPL_2012-06-21_orderbook.csv',
    n_levels=10,  # Nombre de niveaux de prix
    start_time="09:30:00",  # Heure de début (optionnel)
    end_time="16:00:00",    # Heure de fin (optionnel)
    normalize=True          # Normaliser les données
)

# Charger les données
data = loader.load()

# Afficher les premières lignes
print(data.head())
```

### Chargement de Données Binance

```python
from src.data.lob_loader import create_binance_loader

# Créer le loader
loader = create_binance_loader(
    data_path='data/binance/',  # Répertoire contenant les fichiers JSON
    symbol='BTCUSDT',
    n_levels=10,
    normalize=True
)

# Charger les données
data = loader.load()

# Afficher les premières lignes
print(data.head())
```

### Fonctions Utilitaires

```python
# Obtenir la série de prix moyens
midprice = loader.get_midprice_series()

# Obtenir la série de spreads
spread = loader.get_spread_series()

# Calculer les changements de prix
price_changes = loader.get_price_changes(window=1)

# Calculer la volatilité
volatility = loader.get_volatility(window=100)

# Calculer le déséquilibre du flux d'ordres
ofi = loader.get_order_flow_imbalance()

# Calculer la profondeur du marché
depth = loader.get_market_depth(n_levels=5)

# Obtenir un snapshot du carnet d'ordres
snapshot = loader.get_orderbook_snapshot(index=0)
print(snapshot)
```

---

## Environnement RL

### Création de l'Environnement

```python
from src.rl_env.lob_market_making_env import create_lob_env

# Pour LOBSTER
env = create_lob_env(
    data_source='lobster',
    message_file='data/AAPL_message.csv',
    orderbook_file='data/AAPL_orderbook.csv',
    episode_length=1000,  # Nombre de pas par épisode
    reward_type='pnl',     # Type de récompense
    seed=42
)

# Pour Binance
env = create_lob_env(
    data_source='binance',
    data_path='data/binance/',
    symbol='BTCUSDT',
    episode_length=1000,
    reward_type='pnl',
    seed=42
)
```

### Configuration Avancée

```python
from src.rl_env.lob_market_making_env import LOBMarketMakingEnvConfig

config = LOBMarketMakingEnvConfig(
    # Données
    data_source='lobster',
    message_file='data/AAPL_message.csv',
    orderbook_file='data/AAPL_orderbook.csv',
    
    # LOB
    n_levels=10,
    tick_size=0.01,
    
    # Épisode
    episode_length=1000,
    start_idx=0,
    
    # Inventaire
    Q_max=10,
    
    # Actions (spreads)
    delta_min=0.001,
    delta_max=0.1,
    n_spread_levels=10,
    
    # Récompense
    reward_type='pnl',  # 'pnl', 'sharpe', 'inventory_penalty', 'spread_profit'
    inventory_penalty=0.01,
    transaction_cost=0.0001,
    
    # Features
    use_lob_features=True,
    use_order_flow=True,
    use_volatility=True,
    volatility_window=100,
    
    seed=42
)

env = LOBMarketMakingEnv(config)
```

### Espace d'Observation

L'observation inclut :
- `time` : Temps normalisé (0 à 1)
- `inventory` : Inventaire normalisé (-1 à 1)
- `price` : Prix moyen normalisé
- `price_change` : Changement de prix
- `spread` : Spread normalisé (si `use_lob_features=True`)
- `market_depth` : Profondeur du marché (si `use_lob_features=True`)
- `ofi` : Déséquilibre du flux d'ordres (si `use_order_flow=True`)
- `volatility` : Volatilité (si `use_volatility=True`)

### Espace d'Action

L'action est discrète : `action = bid_level * n_spread_levels + ask_level`

- `bid_level` : Niveau de spread bid (0 à `n_spread_levels-1`)
- `ask_level` : Niveau de spread ask (0 à `n_spread_levels-1`)

### Utilisation de l'Environnement

```python
# Réinitialiser l'environnement
obs, info = env.reset(seed=42)

# Exécuter un pas
action = env.action_space.sample()  # Action aléatoire
obs, reward, terminated, truncated, info = env.step(action)

# Rendre l'environnement
env.render(mode='human')

# Fermer l'environnement
env.close()
```

---

## Entraînement d'un Agent RL

### Entraînement avec le Script

```bash
# Entraînement avec LOBSTER
python experiments/train_rl_lob.py \
    --source lobster \
    --message-file data/AAPL_message.csv \
    --orderbook-file data/AAPL_orderbook.csv \
    --algorithm ppo \
    --timesteps 100000 \
    --reward-type pnl

# Entraînement avec Binance
python experiments/train_rl_lob.py \
    --source binance \
    --data-path data/binance/ \
    --symbol BTCUSDT \
    --algorithm ppo \
    --timesteps 100000 \
    --reward-type pnl
```

### Algorithmes Disponibles

- **PPO** (Proximal Policy Optimization) : Recommandé pour la plupart des cas
- **DQN** (Deep Q-Network) : Pour les espaces d'action discrets
- **SAC** (Soft Actor-Critic) : Pour les environnements stochastiques
- **A2C** (Advantage Actor-Critic) : Alternative à PPO

### Types de Récompense

- **`pnl`** : Changement de P&L
- **`sharpe`** : Ratio de Sharpe des rendements récents
- **`inventory_penalty`** : P&L moins pénalité d'inventaire
- **`spread_profit`** : Profit de spread lors des trades

### Entraînement en Python

```python
from stable_baselines3 import PPO
from src.rl_env.lob_market_making_env import create_lob_env

# Créer l'environnement
env = create_lob_env(
    data_source='lobster',
    message_file='data/AAPL_message.csv',
    orderbook_file='data/AAPL_orderbook.csv',
    episode_length=1000,
    reward_type='pnl'
)

# Créer le modèle
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95
)

# Entraîner
model.learn(total_timesteps=100000)

# Sauvegarder
model.save("models/ppo_lobster")
```

### Callbacks Personnalisés

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
    
    def _on_step(self):
        # Logique personnalisée
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
        return True

# Utiliser le callback
callback = CustomCallback(verbose=1)
model.learn(total_timesteps=100000, callback=callback)
```

---

## Évaluation

### Évaluation avec le Script

```bash
# Évaluer un modèle entraîné
python experiments/train_rl_lob.py \
    --mode evaluate \
    --model models/rl_lob_model.zip \
    --source lobster \
    --message-file data/AAPL_message.csv \
    --orderbook-file data/AAPL_orderbook.csv \
    --n-episodes 10 \
    --render
```

### Évaluation en Python

```python
from stable_baselines3 import PPO
from src.rl_env.lob_market_making_env import create_lob_env

# Charger le modèle
model = PPO.load("models/ppo_lobster")

# Créer l'environnement
env = create_lob_env(
    data_source='lobster',
    message_file='data/AAPL_message.csv',
    orderbook_file='data/AAPL_orderbook.csv'
)

# Évaluer
episode_rewards = []
episode_pnls = []

for episode in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    episode_rewards.append(total_reward)
    episode_pnls.append(info['pnl'])
    print(f"Episode {episode}: Reward={total_reward:.2f}, PnL={info['pnl']:.2f}")

# Résultats
print(f"Mean Reward: {np.mean(episode_rewards):.2f}")
print(f"Mean PnL: {np.mean(episode_pnls):.2f}")
```

---

## Exemples Pratiques

### Exemple 1 : Entraînement Complet avec LOBSTER

```python
import numpy as np
from stable_baselines3 import PPO
from src.data.lob_loader import create_lobster_loader
from src.rl_env.lob_market_making_env import create_lob_env

# 1. Charger et explorer les données
loader = create_lobster_loader(
    message_file='data/AAPL_message.csv',
    orderbook_file='data/AAPL_orderbook.csv',
    n_levels=10
)
data = loader.load()

print(f"Data shape: {data.shape}")
print(f"Price range: {data['midprice'].min():.2f} - {data['midprice'].max():.2f}")
print(f"Mean spread: {data['spread'].mean():.4f}")

# 2. Créer l'environnement
env = create_lob_env(
    data_source='lobster',
    message_file='data/AAPL_message.csv',
    orderbook_file='data/AAPL_orderbook.csv',
    episode_length=1000,
    reward_type='pnl',
    seed=42
)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# 3. Créer et entraîner le modèle
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

model.learn(total_timesteps=100000)

# 4. Sauvegarder le modèle
model.save("models/ppo_aapl_lobster")

# 5. Évaluer
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Final PnL: {info['pnl']:.2f}")
print(f"Total Reward: {total_reward:.2f}")
print(f"Trades: {info['n_trades']}")
```

### Exemple 2 : Comparaison de Stratégies

```python
from stable_baselines3 import PPO, DQN
from src.rl_env.lob_market_making_env import create_lob_env

# Créer l'environnement
env = create_lob_env(
    data_source='binance',
    data_path='data/binance/',
    symbol='BTCUSDT',
    episode_length=1000,
    reward_type='pnl'
)

# Entraîner PPO
ppo_model = PPO("MlpPolicy", env, verbose=0)
ppo_model.learn(total_timesteps=50000)

# Entraîner DQN
dqn_model = DQN("MlpPolicy", env, verbose=0)
dqn_model.learn(total_timesteps=50000)

# Comparer
def evaluate(model, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

ppo_mean, ppo_std = evaluate(ppo_model)
dqn_mean, dqn_std = evaluate(dqn_model)

print(f"PPO: {ppo_mean:.2f} ± {ppo_std:.2f}")
print(f"DQN: {dqn_mean:.2f} ± {dqn_std:.2f}")
```

### Exemple 3 : Hyperparameter Tuning

```python
from stable_baselines3 import PPO
from src.rl_env.lob_market_making_env import create_lob_env

env = create_lob_env(
    data_source='lobster',
    message_file='data/AAPL_message.csv',
    orderbook_file='data/AAPL_orderbook.csv',
    episode_length=1000
)

# Grille d'hyperparamètres
learning_rates = [1e-4, 3e-4, 1e-3]
batch_sizes = [32, 64, 128]

results = []

for lr in learning_rates:
    for bs in batch_sizes:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=lr,
            batch_size=bs,
            n_steps=2048
        )
        
        model.learn(total_timesteps=20000)
        
        # Évaluation rapide
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        results.append({
            'lr': lr,
            'batch_size': bs,
            'reward': total_reward
        })
        print(f"lr={lr}, bs={bs}: reward={total_reward:.2f}")

# Meilleurs hyperparamètres
best = max(results, key=lambda x: x['reward'])
print(f"\nBest: lr={best['lr']}, batch_size={best['batch_size']}")
```

---

## Dépannage

### Erreur : "No JSON files found in data_path"

**Cause :** Le répertoire spécifié ne contient pas de fichiers JSON.

**Solution :** Vérifiez le chemin et assurez-vous que les fichiers JSON de Binance sont présents.

### Erreur : "stable-baselines3 not installed"

**Cause :** La bibliothèque stable-baselines3 n'est pas installée.

**Solution :** 
```bash
pip install stable-baselines3
```

### Erreur : "IndexError: list index out of range"

**Cause :** L'épisode dépasse la fin des données disponibles.

**Solution :** Réduisez `episode_length` ou augmentez la taille de vos données.

### Performance Lente

**Solutions :**
1. Réduisez `n_levels` dans la configuration
2. Désactivez certaines features (`use_lob_features=False`, `use_volatility=False`)
3. Utilisez un environnement vectorisé (`DummyVecEnv` ou `SubprocVecEnv`)
4. Réduisez `episode_length`

### Overfitting

**Solutions :**
1. Utilisez plus de données d'entraînement
2. Augmentez la régularisation (`ent_coef` dans PPO)
3. Utilisez la validation croisée
4. Réduisez la complexité du modèle

---

## Ressources Supplémentaires

- [Documentation Gymnasium](https://gymnasium.farama.org/)
- [Documentation Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [LOBSTER Data](https://lobsterdata.com/)
- [Binance API](https://binance-docs.github.io/apidocs/)

---

## Support

Pour toute question ou problème, n'hésitez pas à consulter :
- Le README principal du projet
- La documentation mathématique (`docs/math_model.md`)
- Les exemples dans le répertoire `experiments/`
