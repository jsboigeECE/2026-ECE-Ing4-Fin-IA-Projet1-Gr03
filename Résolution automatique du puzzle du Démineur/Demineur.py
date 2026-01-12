import random
import numpy as np

def creation_demineur(taille, bombes):
    demineur = np.zeros((taille, taille), dtype=int)

    i = 0
    while i < bombes:
        x = random.randint(0, taille-1)
        y = random.randint(0, taille-1)

        if demineur[x][y] == 0:
            #Le niméro 9 représente une mine
            demineur[x][y] = 9
            i+=1


    return demineur

def ajout_nbr_casevide(demineur, i, j):
    if demineur[i][j] != 9:
        demineur[i][j]+=1

def mine_adjacent(demineur, taille):

    for i in range(taille):
        for j in range(taille):
            if demineur[i][j] == 9:
                if i != 0:
                    if j != 0:
                        ajout_nbr_casevide(demineur, i-1, j-1)
                    ajout_nbr_casevide(demineur, i-1, j)
                    if j != taille-1:
                        ajout_nbr_casevide(demineur, i-1, j+1)
                if j != 0:
                    ajout_nbr_casevide(demineur, i, j-1)
                if j != taille-1:
                    ajout_nbr_casevide(demineur, i, j+1)
                if i != taille-1:
                    if j != 0:
                        ajout_nbr_casevide(demineur, i+1, j-1)
                    ajout_nbr_casevide(demineur, i+1, j)
                    if j != taille-1:
                        ajout_nbr_casevide(demineur, i+1, j+1)
    return demineur


def affichage(demineur, taille, mask):
    for i in range(taille):
        for k in range(taille*4):
            print("_", end='')
            if k == taille*4 - 1:
                print("_", end='\n')

        for j in range(taille):
            if mask[i][j]:
                print("#", end=' | ')
            else:
                print(demineur[i][j], end=' | ')
            if j == taille-1:
                print("", end='\n')

    print()
    print()

def jeu(demineur, mask, x, y):
    if demineur[x][y] == 9:
        print("Perdu")
    mask[x][y] = 0

demineur = creation_demineur(10, 7)
print(demineur)
demineur = mine_adjacent(demineur, 10)
print(demineur)

mask = np.ones(demineur.shape, dtype=int)

affichage(demineur, 10, mask)

jeu(demineur, mask, 3, 3)

affichage(demineur, 10, mask)
