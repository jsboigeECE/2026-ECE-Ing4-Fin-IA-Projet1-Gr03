def compute_optimal_spread(inventory, base_spread=1.0, alpha=0.1):
    """ 
    Calcule un spread optimal simplifié inspiré de l'HJB.
    
    inventory   : inventaire actuel
    base_spread : spread minimal
    alpha       : aversion au risque
    """
    spread = base_spread + alpha * abs(inventory) 
    """ abs = absolute value : permet de mesurer la distance à zéro de l'inventaire. 
    Car plus ton q devient grand =distant de zéro augmente,
    plus ton spread s'élargit 
    (=Plus mon stock s'éloigne de zéro, plus j'augmente mon prix pour me protéger))"""
    return spread

