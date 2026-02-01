# Explication théorique 

**Intro**

 Le market making consiste à fournir de la liquidité en proposant continuellement des prix bid et ask. Le market maker fait face à un problème d'optimisation complexe : maximiser le profit du spread tout en gérant le risque d'inventaire. Ce problème de contrôle stochastique avec contraintes se formule comme un problème HJB (Hamilton-Jacobi-Bellman) discrétisé ou comme CSP dynamique.


 ques-ce que HJB discrétisé ? (L'école Guéant / Lehalle) 

 
 L'équation de Hamilton-Jacobi-Bellman (HJB) est utilisée ici pour trouver la "fonction de valeur", c'est-à-dire le profit futur maximum attendu compte tenu de votre inventaire actuel.
 Ce que l'on discrétise :
  - Le Temps (t) : On découpe la séance de trading en petits intervalles (ex: chaque milliseconde ou chaque seconde).
  - L'Inventaire (q) : 
  C'est crucial. Vous définissez une grille de positions possibles (ex: de -100 à +100 titres).
  - Le Prix (S) : Bien que souvent modélisé par un mouvement brownien, on le discrétise pour le calcul numérique.
  
  L'Objectif : À chaque point de la grille (t, q), l'algorithme calcule le spread optimal (la distance de vos prix bid/ask par rapport au prix moyen) qui maximise votre gain tout en vous ramenant vers un inventaire neutre (q=0).
  
  Outils Python : On utilisera NumPy pour créer ces matrices/grilles et résoudre l'équation par récurrence (en partant de la fin de la journée vers le début).


  Ques-ce qu'est CSP ? 


  Le CSP (Constraint Satisfaction Problem) est particulièrement utile pour intégrer les "contraintes dures" que les modèles mathématiques purs ont parfois du mal à gérer.

  Les Variables : 
  
  - Vos prix bid/ask à chaque instant t.
  - Les Contraintes (Le cœur du CSP) :
  - Position Max : "Interdiction stricte de dépasser Q_{max} titres" (pour éviter la faillite).
  - VaR limite (Value at Risk) : "La perte potentielle ne doit jamais dépasser X euros avec 95% de confiance".
  - Drawdown : "Si la perte de la journée atteint $Y$, on arrête tout".
  
  Pourquoi "Dynamique" ? Car la satisfaction de la contrainte à t=10 dépend de ce que vous avez acheté à t=9.Outil Python : OR-Tools (de Google) est parfait pour cela. 
  Il va chercher, à chaque étape, le meilleur spread qui "respecte" toutes ces contraintes de risque sans jamais les violer.

  Vous l'aurez compris ces deux méthodes sont complémentaires :
  L'une donne la direction (HJB) et l'autre établis les contraintes/limites (CSP).
    