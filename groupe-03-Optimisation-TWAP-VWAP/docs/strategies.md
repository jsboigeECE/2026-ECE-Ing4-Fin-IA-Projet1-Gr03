# Strategies

Ce projet implémente quatre stratégies d’exécution d’ordres : TWAP, VWAP, une optimisation sous contraintes (CP-SAT) et une approche par apprentissage par renforcement (RL).

---

## 1. TWAP (Time-Weighted Average Price)

**Principe**  
Le volume total à exécuter est réparti uniformément dans le temps. TWAP constitue une baseline simple et utile pour comparaison avec des stratégies plus sophistiquées.

**Formule**  
Pour un volume total (Q) et (N) tranches :

$$
x_t = Q / N
$$

(arrondi à l’entier et ajusté pour sommer exactement à (Q))

**Avantages**

* Très simple
* Prévisible

**Limites**

* Ignore totalement la liquidité du marché

---

## 2. VWAP (Volume-Weighted Average Price)

**Principe**  
Le volume est réparti proportionnellement au volume de marché observé ou historique, ce qui permet de tenir compte de la liquidité réelle du marché.

**Formule**  
Pour des volumes de marché (V_t) :

$$
x_t = Q \cdot \frac{V_t}{\sum_t V_t}
$$

Une contrainte de participation limite l’exécution :

$$
x_t \leq \alpha \cdot V_t
$$

avec $$(\alpha) le taux de participation.

**Avantages**

* S’aligne sur la liquidité réelle

**Limites**

* Dépend des volumes observés

---

## 3. Optimisation sous contraintes (CP-SAT)

**Principe**  
Formuler l’exécution comme un problème d’optimisation sous contraintes permet de minimiser l’impact de marché tout en respectant les contraintes imposées.

**Contraintes principales**

$$
\sum_t x_t = Q
$$

$$
0 \leq x_t \leq \alpha \cdot V_t
$$

**Fonction objectif**  
Minimisation d’un compromis entre impact et tracking VWAP :

$$
\min \sum_t \left( \lambda_{\text{impact}} x_t^2 + \lambda_{\text{track}} (x_t - x_t^{\text{VWAP}})^2 \right)
$$

**Avantages**

* Flexible
* Paramétrable

**Limites**

* Suppose une connaissance a posteriori des volumes

---

## 4. Reinforcement Learning (Q-learning)

**Principe**  
Un agent apprend une politique d’exécution par interaction avec un environnement simulé. Cette approche est adaptative et apprend en ligne.

**État**  
(t, q_remaining)

**Action**  
Fraction du volume maximal autorisé à l’instant (t).

**Récompense**

$$
r_t = - \left( \lambda_{\text{impact}} x_t^2 + \lambda_{\text{track}} (x_t - x_t^{\text{VWAP}})^2 \right)
$$

Pénalité terminale si (Q) n’est pas entièrement exécuté.

**Avantages**

* Adaptatif
* Approche en ligne

**Limites**

* Dépend de l’entraînement et des hyperparamètres

---

Ces stratégies sont comparées sur les mêmes métriques afin d’analyser le compromis entre impact de marché et suivi du VWAP.
