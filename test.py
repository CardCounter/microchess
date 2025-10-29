import pygame
import sys
import numpy as np
from game import MicrochessEnv, WHITE, BLACK, BOARD_ROWS, BOARD_COLS, PIECE_WK, PIECE_WR, PIECE_WB, PIECE_WN, PIECE_WP

TILE_SIZE = 100
PADDING = 20
WIDTH = BOARD_COLS * TILE_SIZE
HEIGHT = BOARD_ROWS * TILE_SIZE + PADDING

BG_LIGHT = (230, 230, 230)
BG_DARK  = (180, 180, 180)
HIGHLIGHT = (255, 255, 0)
ILLEGAL_COLOR = (255, 50, 50)
TEXT_COLOR = (20, 20, 20)
WHITE_PIECE_COLOR = (240, 240, 255)
BLACK_PIECE_COLOR = (40, 40, 60)

pygame.init()
BIGFONT = pygame.font.SysFont("arial", 32, bold=True)
PIECEFONT = pygame.font.SysFont("arial", max(32, TILE_SIZE - 20), bold=True)

# Helper to check if font can render a glyph
def font_has_glyph(font, text):
    # pygame.font.Font.render will draw tofu boxes if missing glyph.
    # We can detect missing glyphs by checking if the rendered surface is basically empty width-wise.
    # Simpler: try render and see if resulting surface width > 0 for non-empty text.
    if text == "":
        return True
    surf = font.render(text, True, (0,0,0))
    return surf.get_width() > 0

def piece_to_str(p):
    """
    returns glyph we try to draw for this piece.
    first try unicode chess glyphs.
    if that fails to render (font missing glyph), we'll still draw the returned string,
    but it'll be a simple letter like 'K','Q','R','B','N','P' or lowercase for black.
    """
    if p == 0:
        return ""
    is_white = (np.sign(p) == WHITE)
    t = abs(p)

    # mapping for fallback ASCII
    fallback = {
        PIECE_WK: ("K","k"),
        PIECE_WR: ("R","r"),
        PIECE_WB: ("B","b"),
        PIECE_WN: ("N","n"),
        PIECE_WP: ("P","p"),
    }

    unicode_map = {
        PIECE_WK: ("♔","♚"),
        PIECE_WR: ("♖","♜"),
        PIECE_WB: ("♗","♝"),
        PIECE_WN: ("♘","♞"),
        PIECE_WP: ("♙","♟"),
    }

    # choose unicode symbol for this type
    if t in unicode_map:
        uni_white, uni_black = unicode_map[t]
        letter_white, letter_black = fallback[t]
        return uni_white if is_white else uni_black

    # if somehow unknown piece code
    # fall back to ascii anyway
    if t in fallback:
        letter_white, letter_black = fallback[t]
        return letter_white if is_white else letter_black
    return "?"

def generate_all_legal_map(env):
    """
    returns:
    - legal_moves: list of full move dicts from env.generate_legal_moves
    - moves_by_from[(r,c)] = list of moves starting from that square
    """
    legal_moves = env.generate_legal_moves(env.side_to_move)
    moves_by_from = {}
    for m in legal_moves:
        fr, fc = m["from"]
        moves_by_from.setdefault((fr,fc), []).append(m)
    return legal_moves, moves_by_from

def find_move(moves, fr, fc, tr, tc, promo_choice=None):
    """
    match a move in moves with given coords.
    promo_choice can be 0/1/2 if we want specific promotion.
    if no promotion, accept the non-promo move.
    """
    for m in moves:
        if m["from"] == (fr,fc) and m["to"] == (tr,tc):
            if m["promotion"]:
                # if promotion, check promo type
                if promo_choice is None or m["promo_type"] == promo_choice:
                    return m
            else:
                # no promotion. allow as long as promo_choice isn't forcing something
                if promo_choice is None:
                    return m
    return None

def draw_board(screen, env, selected_sq, legal_targets, game_over, result):
    screen.fill((0,0,0))
    board = env.board

    # draw tiles
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(c*TILE_SIZE, r*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            base_col = BG_LIGHT if (r+c)%2==0 else BG_DARK

            # highlight selected origin
            if selected_sq == (r,c):
                pygame.draw.rect(screen, HIGHLIGHT, rect)
            else:
                pygame.draw.rect(screen, base_col, rect)

            # highlight legal targets
            if (r,c) in legal_targets:
                # overlay a ring
                pygame.draw.rect(screen, (255, 255, 0), rect, 4)

            # draw piece as white/black circle with letter
            p = board[r,c]
            if p != 0:
                is_white = np.sign(p) == WHITE
                piece_col = WHITE_PIECE_COLOR if is_white else BLACK_PIECE_COLOR
                pygame.draw.circle(screen, piece_col, rect.center, TILE_SIZE//2 - 10)

                # determine letter symbol
                t = abs(p)
                symbol_map = {
                    PIECE_WK: "K",
                    PIECE_WR: "R",
                    PIECE_WB: "B",
                    PIECE_WN: "N",
                    PIECE_WP: "P",
                }
                label = symbol_map.get(t, "?")
                text_surf = PIECEFONT.render(label, True, TEXT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)

    # draw footer area
    footer_rect = pygame.Rect(0, BOARD_ROWS*TILE_SIZE, WIDTH, PADDING)
    pygame.draw.rect(screen, (30,30,30), footer_rect)

    if not game_over:
        stm_text = "White to move" if env.side_to_move == WHITE else "Black to move"
    else:
        if result == 1:
            stm_text = "White wins"
        elif result == -1:
            stm_text = "Black wins"
        else:
            stm_text = "Draw"

    stm_surf = BIGFONT.render(stm_text, True, (255,255,255))
    stm_rect = stm_surf.get_rect(midleft=(10, BOARD_ROWS*TILE_SIZE + PADDING//2))
    screen.blit(stm_surf, stm_rect)

    pygame.display.flip()

def main():
    env = MicrochessEnv()
    obs = env.reset()

    selected_sq = None  # (r,c) of piece we're trying to move
    cached_legal_moves, moves_by_from = generate_all_legal_map(env)
    game_over = False
    terminal_result = None

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Microchess")

    while True:
        clock.tick(60)

        # figure out which squares should be outlined as allowed targets if something is selected
        legal_targets = set()
        if selected_sq is not None and not game_over:
            moves_for_sq = moves_by_from.get(selected_sq, [])
            for m in moves_for_sq:
                legal_targets.add(m["to"])

        draw_board(screen, env, selected_sq, legal_targets, game_over, terminal_result)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game_over:
                    # click after game over = reset
                    env.reset()
                    selected_sq = None
                    cached_legal_moves, moves_by_from = generate_all_legal_map(env)
                    game_over = False
                    terminal_result = None
                    continue

                mx, my = event.pos
                r = my // TILE_SIZE
                c = mx // TILE_SIZE

                # in case you click below board (footer), ignore
                if r < 0 or r >= BOARD_ROWS or c < 0 or c >= BOARD_COLS:
                    continue

                if selected_sq is None:
                    # first click: select a piece IF it's the side to move
                    p = env.board[r,c]
                    if p != 0 and np.sign(p) == env.side_to_move:
                        selected_sq = (r,c)
                else:
                    # second click: try to move from selected_sq -> (r,c)
                    fr, fc = selected_sq
                    tr, tc = r, c

                    # is that legal?
                    move = find_move(cached_legal_moves, fr, fc, tr, tc)
                    if move is not None:
                        # apply
                        env._apply_move(move)

                        # check terminal
                        done, result = env._check_terminal()
                        if done:
                            game_over = True
                            terminal_result = result
                        else:
                            # still going, recompute legal moves
                            cached_legal_moves, moves_by_from = generate_all_legal_map(env)

                        # clear selection either way
                        selected_sq = None
                    else:
                        # clicked somewhere illegal -> either reselect same-color piece or clear
                        p2 = env.board[r,c]
                        if p2 != 0 and np.sign(p2) == env.side_to_move:
                            selected_sq = (r,c)  # switch selection to new piece
                        else:
                            selected_sq = None

if __name__ == "__main__":
    main()