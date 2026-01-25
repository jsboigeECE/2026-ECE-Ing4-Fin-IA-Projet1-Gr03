# Sujet 50 

# Farhan et Ilhan


1) Market making en ~10 lignes (bid/ask, spread, exécutions, inventaire)

Le market maker fournit de la liquidité en affichant en continu deux prix : un bid (prix d’achat) et un ask (prix de vente).
La différence entre ask et bid est le spread, qui représente la marge potentielle capturée si le market maker achète au bid puis revend au ask.
Quand un autre participant envoie un ordre “au marché”, il peut taper votre bid (vous achetez) ou votre ask (vous vendez) : ce sont les exécutions.
Après exécution, votre inventaire (position) change : si vous achetez, votre inventaire augmente ; si vous vendez, il diminue.
Votre profit ne vient pas seulement du spread : votre PnL dépend aussi de la variation du prix (mark-to-market) sur l’inventaire que vous détenez.
Si le prix bouge contre votre position (ex : vous êtes long et le prix baisse), vous subissez une perte.
Le market maker doit donc ajuster ses quotes (leur niveau et leur asymétrie) pour continuer à être exécuté tout en maîtrisant son exposition.
En pratique, il “skew” ses prix : s’il est trop long, il rend son ask plus attractif (pour vendre) et son bid moins attractif (pour éviter d’acheter).
Le problème est dynamique (le marché bouge, les exécutions sont aléatoires) et se formule en contrôle stochastique.
L’objectif : fournir des prix compétitifs, gagner le spread, et contrôler le risque lié à l’inventaire.

2) Le trade-off “profit du spread” vs “risque d’inventaire”

Si on met un spread large : on gagne plus par trade si on est exécuté, mais on est moins souvent exécuté (moins de volume, moins de profits).

Si on met un spread serré : on est exécuté plus souvent, mais la marge par trade est plus faible, et on peut accumuler vite un inventaire important.

Le vrai danger vient de l’inventaire :

plus l’inventaire |q| est grand, plus le PnL devient sensible aux mouvements du prix (risque “directionnel” non désiré),

donc on doit parfois sacrifier du profit (en modifiant le spread/skew ou en stoppant certaines quotes) pour réduire l’inventaire.

En résumé :
👉 maximiser le gain du spread pousse à coter agressif et être exécuté,
👉 minimiser le risque d’inventaire pousse à contrôler |q| via des quotes asymétriques, des contraintes (q max, VaR proxy), ou une liquidation.