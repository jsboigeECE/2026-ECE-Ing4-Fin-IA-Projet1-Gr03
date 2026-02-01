def compute_optimal_spread(inventory, base_spread=1.0, alpha=0.1):
    """ 
    Calcule un spread optimal simplifié inspiré de l'HJB.
    
    inventory   : inventaire actuel
    base_spread : spread minimal
    alpha       : aversion au risque
    """
    spread = base_spread + alpha * abs(inventory)
    return spread
