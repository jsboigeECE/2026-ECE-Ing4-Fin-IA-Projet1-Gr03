"""import pygame
import sys

# Constantes graphiques
CELL_SIZE = 40           # Taille d’une case en pixels
MARGIN = 2               # Petit espace entre les cases
TOP_BAR = 60             # Espace en haut pour le titre / infos

# Couleurs
BG_COLOR = (30, 30, 30)
GRID_BG = (50, 50, 50)
HIDDEN_COLOR = (120, 120, 120)
REVEALED_COLOR = (200, 200, 200)
MINE_COLOR = (200, 50, 50)
TEXT_COLOR = (20, 20, 20)
WIN_TEXT_COLOR = (50, 200, 50)
LOSE_TEXT_COLOR = (220, 50, 50)

# Couleurs pour les chiffres (1–8)
NUMBER_COLORS = {
    1: (25, 118, 210),
    2: (56, 142, 60),
    3: (211, 47, 47),
    4: (123, 31, 162),
    5: (255, 143, 0),
    6: (0, 151, 167),
    7: (85, 139, 47),
    8: (66, 66, 66),
}


def init_pygame(taille):
    pygame.init()
    width = taille * (CELL_SIZE + MARGIN) + MARGIN
    height = TOP_BAR + taille * (CELL_SIZE + MARGIN) + MARGIN
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Démineur CSP")

    font_cell = pygame.font.SysFont("consolas", 24, bold=True)
    font_title = pygame.font.SysFont("consolas", 28, bold=True)
    return screen, font_cell, font_title


def draw_grid(screen, font_cell, font_title, demineur, mask, taille, game_state):

    screen.fill(BG_COLOR)

    # Titre / état
    if game_state == "running":
        text = font_title.render("Démineur - En cours", True, (230, 230, 230))
    elif game_state == "win":
        text = font_title.render("Gagné ! (R pour rejouer)", True, WIN_TEXT_COLOR)
    else:
        text = font_title.render("Perdu... (R pour rejouer)", True, LOSE_TEXT_COLOR)

    screen.blit(text, (10, 10))

    # Fond de la grille
    grid_rect = pygame.Rect(0, TOP_BAR, screen.get_width(), screen.get_height() - TOP_BAR)
    pygame.draw.rect(screen, GRID_BG, grid_rect)

    # Cases
    for i in range(taille):
        for j in range(taille):
            x = MARGIN + j * (CELL_SIZE + MARGIN)
            y = TOP_BAR + MARGIN + i * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            if mask[i][j] == 1:  # cachée
                pygame.draw.rect(screen, HIDDEN_COLOR, rect)
            else:  # révélée
                if demineur[i][j] == 9:
                    pygame.draw.rect(screen, MINE_COLOR, rect)
                    # petit cercle noir pour symboliser la mine
                    pygame.draw.circle(
                        screen,
                        (0, 0, 0),
                        (x + CELL_SIZE // 2, y + CELL_SIZE // 2),
                        CELL_SIZE // 4,
                    )
                else:
                    pygame.draw.rect(screen, REVEALED_COLOR, rect)
                    val = demineur[i][j]
                    if val > 0:
                        color = NUMBER_COLORS.get(val, TEXT_COLOR)
                        txt = font_cell.render(str(val), True, color)
                        txt_rect = txt.get_rect(center=rect.center)
                        screen.blit(txt, txt_rect)

    pygame.display.flip()


def coords_from_mouse(pos, taille):
    # Convertit une position de souris (x, y) en indices (i, j) dans la grille, ou None si hors grille
    x, y = pos
    # On enlève le TOP_BAR pour la coordonnée verticale
    y_grid = y - TOP_BAR
    if y_grid < 0:
        return None

    # Pour chaque colonne / ligne
    for i in range(taille):
        for j in range(taille):
            cell_x = MARGIN + j * (CELL_SIZE + MARGIN)
            cell_y = MARGIN + i * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
            rect.move_ip(0, TOP_BAR)  # décalage vertical
            if rect.collidepoint(x, y):
                return i, j

    return None


def pygame_game_loop(demineur, mask, taille, bombes, jeu_func, fin_jeu_func, restart_callback):

    screen, font_cell, font_title = init_pygame(taille)
    clock = pygame.time.Clock()

    game_state = "running"  # "running", "win", "lose"

    running = True
    while running:
        clock.tick(60)

        # Si la partie est encore en cours, on teste la fin
        if game_state == "running":
            status = fin_jeu_func(demineur, mask, taille, bombes)
            if status == 0:
                # On regarde si c'est gagné ou perdu en inspectant le plateau
                perdu = False
                for i in range(taille):
                    for j in range(taille):
                        if demineur[i][j] == 9 and mask[i][j] == 0:
                            perdu = True
                            break
                    if perdu:
                        break
                game_state = "lose" if perdu else "win"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Clic gauche pour révéler une case si la partie est en cours
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and game_state == "running":
                grid_pos = coords_from_mouse(event.pos, taille)
                if grid_pos is not None:
                    i, j = grid_pos
                    # Appel à la fonction de jeu existante
                    jeu_func(demineur, mask, taille, i, j)

            # Touche R pour recommencer après fin de partie
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_state in ("win", "lose"):
                demineur, mask = restart_callback()
                game_state = "running"

        draw_grid(screen, font_cell, font_title, demineur, mask, taille, game_state)

    pygame.quit()
    sys.exit()"""


import pygame
import sys
import time

CELL_SIZE = 42
MARGIN = 4
TOP_BAR = 80

# Couleurs
BG_TOP = (30, 30, 60)
BG_BOTTOM = (10, 10, 25)
GRID_BG = (40, 40, 60)

HIDDEN_COLOR = (70, 80, 110)
HIDDEN_HOVER = (90, 100, 135)
REVEALED_COLOR = (215, 220, 240)
MINE_COLOR = (220, 80, 80)
MINE_CENTER = (40, 0, 0)

TEXT_COLOR = (20, 20, 40)
WIN_TEXT_COLOR = (120, 230, 150)
LOSE_TEXT_COLOR = (240, 90, 90)

NUMBER_COLORS = {
    1: (25, 118, 210),
    2: (56, 142, 60),
    3: (211, 47, 47),
    4: (123, 31, 162),
    5: (255, 143, 0),
    6: (0, 151, 167),
    7: (85, 139, 47),
    8: (66, 66, 66),
}


def draw_vertical_gradient(surface, top_color, bottom_color):
    """Fond dégradé vertical simple."""
    width, height = surface.get_size()
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        pygame.draw.line(surface, (r, g, b), (0, y), (width, y))


def init_pygame(taille):
    pygame.init()
    width = taille * (CELL_SIZE + MARGIN) + MARGIN
    height = TOP_BAR + taille * (CELL_SIZE + MARGIN) + MARGIN
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Démineur CSP")

    font_cell = pygame.font.SysFont("consolas", 24, bold=True)
    font_title = pygame.font.SysFont("consolas", 30, bold=True)
    font_small = pygame.font.SysFont("consolas", 20, bold=False)
    return screen, font_cell, font_title, font_small


def draw_top_bar(screen, font_title, font_small, game_state, bombes, demineur, mask, start_time):
    # Fond de la barre
    bar_rect = pygame.Rect(0, 0, screen.get_width(), TOP_BAR)
    pygame.draw.rect(screen, (0, 0, 0, 120), bar_rect)

    # Titre / état
    if game_state == "running":
        title_text = "Démineur CSP"
        color = (230, 230, 240)
    elif game_state == "win":
        title_text = "GAGNÉ ! (R pour rejouer)"
        color = WIN_TEXT_COLOR
    else:
        title_text = "PERDU... (R pour rejouer)"
        color = LOSE_TEXT_COLOR

    title_surface = font_title.render(title_text, True, color)
    screen.blit(title_surface, (15, 10))

    # Compteur de bombes restantes (approx : bombes - cases non révélées qui sont des mines)
    total_mines = bombes
    flagged_like = 0  # tu pourras utiliser de vrais drapeaux plus tard
    mines_text = f"Bombes : {total_mines - flagged_like:02d}"
    mines_surface = font_small.render(mines_text, True, (230, 230, 240))
    screen.blit(mines_surface, (15, 45))

    # Timer
    if start_time is not None:
        elapsed = int(time.time() - start_time)
    else:
        elapsed = 0
    timer_text = f"Temps : {elapsed:03d}s"
    timer_surface = font_small.render(timer_text, True, (230, 230, 240))
    screen.blit(timer_surface, (screen.get_width() - timer_surface.get_width() - 15, 45))


def draw_cell(surface, rect, state, value=None, font=None, hovered=False):
    """
    state: 'hidden', 'revealed', 'mine'
    """
    border_radius = 8

    if state == "hidden":
        color = HIDDEN_HOVER if hovered else HIDDEN_COLOR
        pygame.draw.rect(surface, color, rect, border_radius=border_radius)
        # léger contour
        pygame.draw.rect(surface, (30, 30, 40), rect, width=2, border_radius=border_radius)
    elif state == "revealed":
        pygame.draw.rect(surface, REVEALED_COLOR, rect, border_radius=border_radius)
        pygame.draw.rect(surface, (160, 160, 180), rect, width=1, border_radius=border_radius)
        if value is not None and value > 0 and font is not None:
            color = NUMBER_COLORS.get(value, TEXT_COLOR)
            txt = font.render(str(value), True, color)
            txt_rect = txt.get_rect(center=rect.center)
            surface.blit(txt, txt_rect)
    elif state == "mine":
        pygame.draw.rect(surface, MINE_COLOR, rect, border_radius=border_radius)
        pygame.draw.rect(surface, (120, 20, 20), rect, width=2, border_radius=border_radius)
        pygame.draw.circle(
            surface,
            MINE_CENTER,
            (rect.x + rect.w // 2, rect.y + rect.h // 2),
            rect.w // 4,
        )


def draw_grid(screen, font_cell, font_title, font_small, demineur, mask, taille, game_state, bombes, start_time):
    mouse_pos = pygame.mouse.get_pos()

    # Fond dégradé
    draw_vertical_gradient(screen, BG_TOP, BG_BOTTOM)

    # Fond de la zone de grille
    grid_rect = pygame.Rect(
        0,
        TOP_BAR,
        screen.get_width(),
        screen.get_height() - TOP_BAR,
    )
    pygame.draw.rect(screen, GRID_BG, grid_rect, border_radius=16)

    draw_top_bar(screen, font_title, font_small, game_state, bombes, demineur, mask, start_time)

    for i in range(taille):
        for j in range(taille):
            x = MARGIN + j * (CELL_SIZE + MARGIN)
            y = TOP_BAR + MARGIN + i * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            hovered = rect.collidepoint(mouse_pos)

            if mask[i][j] == 1:
                state = "hidden"
                draw_cell(screen, rect, state, hovered=hovered)
            else:
                if demineur[i][j] == 9:
                    state = "mine"
                    draw_cell(screen, rect, state)
                else:
                    state = "revealed"
                    draw_cell(screen, rect, state, value=demineur[i][j], font=font_cell)

    pygame.display.flip()


def coords_from_mouse(pos, taille):
    x, y = pos
    y_grid = y - TOP_BAR
    if y_grid < 0:
        return None

    for i in range(taille):
        for j in range(taille):
            cell_x = MARGIN + j * (CELL_SIZE + MARGIN)
            cell_y = MARGIN + i * (CELL_SIZE + MARGIN)
            rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
            rect.move_ip(0, TOP_BAR)
            if rect.collidepoint(x, y):
                return i, j

    return None


def pygame_game_loop(demineur, mask, taille, bombes, jeu_func, fin_jeu_func, restart_callback):
    screen, font_cell, font_title, font_small = init_pygame(taille)
    clock = pygame.time.Clock()

    game_state = "running"
    start_time = time.time()

    running = True
    while running:
        clock.tick(60)

        if game_state == "running":
            status = fin_jeu_func(demineur, mask, taille, bombes)
            if status == 0:
                perdu = False
                for i in range(taille):
                    for j in range(taille):
                        if demineur[i][j] == 9 and mask[i][j] == 0:
                            perdu = True
                            break
                    if perdu:
                        break
                game_state = "lose" if perdu else "win"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and game_state == "running":
                grid_pos = coords_from_mouse(event.pos, taille)
                if grid_pos is not None:
                    i, j = grid_pos
                    jeu_func(demineur, mask, taille, i, j)

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_state in ("win", "lose"):
                demineur, mask = restart_callback()
                game_state = "running"
                start_time = time.time()

        draw_grid(
            screen,
            font_cell,
            font_title,
            font_small,
            demineur,
            mask,
            taille,
            game_state,
            bombes,
            start_time,
        )

    pygame.quit()
    sys.exit()