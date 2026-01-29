# Experiments

Cette section décrit les expériences menées pour comparer les différentes stratégies d’exécution et analyser leurs performances dans des conditions variées.

## Stratégies comparées

Les stratégies évaluées comprennent :

- **TWAP** (répartit uniformément le volume sur le temps)
- **VWAP** (répartit proportionnellement aux volumes de marché)
- **Optimisation sous contraintes (CP-SAT)** (compromis entre impact de marché et tracking error)
- **Agent de reinforcement learning (RL)** (apprend et décide des volumes à exécuter tranche par tranche)

## Scénarios

Les tests sont réalisés sur différents types de profils de volume :

- **Profils synthétiques** : générés pour tester des comportements extrêmes et des situations contrôlées.
- **Profils intraday réels** : volumes de marché réels afin d’évaluer les performances dans des conditions réalistes.

## Métriques d’évaluation

### Impact de marché

L’impact de marché est approximé par un **proxy quadratique** du volume exécuté :

$$
Impact = \sum_t x_t^2
$$

où $x_t$ est le volume exécuté à la tranche $t$. Cette métrique pénalise les exécutions agressives.

### Erreur de tracking VWAP

L’erreur de tracking mesure la capacité de la stratégie à suivre le prix moyen pondéré par le volume (VWAP) :

$$
Tracking = \sum_t (x_t - x_t^{VWAP})^2
$$

où $x_t^{VWAP}$ est le volume cible pour suivre le VWAP. Cette métrique quantifie le décalage par rapport au profil de marché.

## Analyse

Les résultats expérimentaux mettent en évidence les caractéristiques et limites de chaque approche :

- **TWAP** : simple à mettre en œuvre, mais rigide, ne tient pas compte de la liquidité.
- **VWAP** : plus aligné sur le marché, mais suppose une connaissance préalable des volumes.
- **CP-SAT** : atteint un compromis optimal entre impact et tracking, respectant les contraintes imposées.
- **RL** : offre une flexibilité adaptative et peut s’ajuster en ligne aux variations du marché.

Ces expériences confirment la cohérence des modèles et permettent de comparer quantitativement les stratégies en fonction de leurs objectifs et contraintes.