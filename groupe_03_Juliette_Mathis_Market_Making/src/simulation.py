import numpy as np
def simulate_price(T=200, S0=100, sigma=1.0):

    """
    Simule l'évolution du prix d'un actif.
    
    T     : nombre de pas de temps
    S0    : prix initial
    sigma : volatilité
    """

    "Liste contenant les prix"
    
    prices = [float(S0)]
    for _ in range (T) : 
        dS = np.random.normal(0,sigma)
        prices.append(prices[-1]+dS)

    return np.array(prices)
    
