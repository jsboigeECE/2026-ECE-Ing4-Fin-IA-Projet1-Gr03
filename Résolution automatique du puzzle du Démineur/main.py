"""import Demineur as d
import numpy as np


def choix_chiffre(max_val, coo):
    while True:
        try:
            n = int(input(f"Choisir une coordonné pour {coo} entre 0 et {max_val} : "))
            if 0 <= n <= max_val:
                break
            else:
                print("❌ Le nombre n’est pas dans la plage.")
        except ValueError:
            print("❌ Entre un nombre valide.")
    return n


taille = 10
bombes = 7

y = choix_chiffre(taille, "x")
x = choix_chiffre(taille, "y")

demineur = d.creation_demineur(taille, bombes, x, y)
print(demineur)
demineur = d.mine_adjacent(demineur, taille)
print(demineur)
mask = np.ones(demineur.shape, dtype=int)
d.jeu(demineur, mask, taille, x, y)
d.affichage(demineur, taille, mask)

while d.fin_jeu(demineur, mask, taille, bombes):
    y = choix_chiffre(taille, "x")
    x = choix_chiffre(taille, "y")

    d.jeu(demineur, mask, taille, x, y)
    d.affichage(demineur, taille, mask)"""

import numpy as np
import Demineur as d
import Affichage


def creer_plateau(taille, bombes):
    # Premier clic fictif au centre pour garantir une zone safe initiale
    x0 = taille // 2
    y0 = taille // 2
    demineur = d.creation_demineur(taille, bombes, x0, y0)
    demineur = d.mine_adjacent(demineur, taille)
    mask = np.ones(demineur.shape, dtype=int)
    # On révèle la première case automatiquement
    d.jeu(demineur, mask, taille, x0, y0)
    return demineur, mask


def main():
    taille = 10
    bombes = 7

    demineur, mask = creer_plateau(taille, bombes)

    def restart_callback():
        return creer_plateau(taille, bombes)

    # Lancement boucle Pygame
    Affichage.pygame_game_loop(
        demineur=demineur,
        mask=mask,
        taille=taille,
        bombes=bombes,
        jeu_func=d.jeu,
        fin_jeu_func=d.fin_jeu,
        restart_callback=restart_callback,
    )


if __name__ == "__main__":
    main()
