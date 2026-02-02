import matplotlib.pyplot as plt

from simulation import simulate_price
from model_hjb import compute_optimal_spread
from strategy import MarketMaker

# On simule les prix
prices = simulate_price()

# On crée le market maker
mm = MarketMaker()

inventories = []
pnls = []

# ⚠️ On commence par stocker le premier prix comme "prix précédent"
prev_price = prices[0]

# On parcourt les prix à partir du deuxième élément
for price in prices[1:]:

    # On calcule le spread optimal
    spread = compute_optimal_spread(mm.inventory)

    # On calcule bid/ask autour du prix précédent
    bid, ask = mm.quote_prices(prev_price, spread)

    # Si le prix actuel descend sous le bid → on achète
    if price < bid:
        mm.execute_order("buy", bid)

    # Si le prix actuel monte au-dessus du ask → on vend
    elif price > ask:
        mm.execute_order("sell", ask)

    # On met à jour les listes
    inventories.append(mm.inventory)
    pnls.append(mm.pnl(price))

    # On met à jour le prix précédent
    prev_price = price

# --- Graphiques ---
plt.figure()
plt.plot(prices)
plt.title("Prix simulé")
plt.show()

plt.figure()
plt.plot(inventories)
plt.title("Inventaire")
plt.show()

plt.figure()
plt.plot(pnls)
plt.title("PnL")
plt.show()
