import numpy as np
from copy import deepcopy

WHITE = 1
BLACK = -1

PIECE_WK = 1
PIECE_WR = 2
PIECE_WB = 3
PIECE_WN = 4
PIECE_WP = 5

PIECE_BK = -1
PIECE_BR = -2
PIECE_BB = -3
PIECE_BN = -4
PIECE_BP = -5

BOARD_ROWS = 5
BOARD_COLS = 4
NUM_SQUARES = BOARD_ROWS * BOARD_COLS

ACTION_BASE = NUM_SQUARES * NUM_SQUARES  # 400
ACTION_SIZE = ACTION_BASE * 3            # 1200
PROMO_NONE_OR_ROOK = 0
PROMO_KNIGHT = 1
PROMO_BISHOP = 2


class MicrochessEnv:
    def __init__(self):
        self.board = None
        self.side_to_move = WHITE
        self.has_moved_king = {WHITE: False, BLACK: False}
        self.has_moved_rook = {WHITE: False, BLACK: False}
        self.move_history = []
        self.reset()

    def reset(self):
        self.board = np.array([
            [PIECE_BK, PIECE_BN, PIECE_BB, PIECE_BR],
            [PIECE_BP, 0,        0,        0],
            [0,        0,        0,        0],
            [0,        0,        0,        PIECE_WP],
            [PIECE_WR, PIECE_WB, PIECE_WN, PIECE_WK]
        ], dtype=np.int8)
        self.side_to_move = WHITE
        self.has_moved_king = {WHITE: False, BLACK: False}
        self.has_moved_rook = {WHITE: False, BLACK: False}
        self.move_history = []
        return self._get_obs()

    def _get_obs(self):
        flat_board = self.board.flatten().astype(np.int8)
        stm = np.int8(self.side_to_move)
        return np.concatenate([flat_board, np.array([stm], dtype=np.int8)], axis=0)

    def step(self, action_id):
        promo_bucket = action_id // ACTION_BASE
        base_id = action_id % ACTION_BASE
        from_sq = base_id // NUM_SQUARES
        to_sq = base_id % NUM_SQUARES
        fr = from_sq // BOARD_COLS
        fc = from_sq % BOARD_COLS
        tr = to_sq // BOARD_COLS
        tc = to_sq % BOARD_COLS

        legal_moves = self.generate_legal_moves(self.side_to_move)
        move_obj = None
        for m in legal_moves:
            if m["from"] == (fr, fc) and m["to"] == (tr, tc):
                if m["promotion"]:
                    if m["promo_type"] == promo_bucket:
                        move_obj = m
                        break
                else:
                    if promo_bucket == 0:
                        move_obj = m
                        break

        if move_obj is None:
            return self._get_obs(), -0.1 * self.side_to_move, False, {"illegal": True}

        self._apply_move(move_obj)
        done, result = self._check_terminal()
        reward = result if done else 0.0
        return self._get_obs(), reward, done, {}

    def generate_legal_moves(self, side):
        pseudo = self._generate_pseudo_legal_moves(side)
        legal = []
        for m in pseudo:
            bcopy = deepcopy(self)
            bcopy._apply_move(m)
            if not bcopy._is_in_check(side):
                legal.append(m)
        return legal

    def _generate_pseudo_legal_moves(self, side):
        moves = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.board[r, c]
                if p == 0 or np.sign(p) != side:
                    continue
                piece_type = abs(p)
                if piece_type == PIECE_WK:
                    moves += self._king_moves(r, c, side)
                elif piece_type == PIECE_WR:
                    moves += self._rook_moves(r, c, side)
                elif piece_type == PIECE_WB:
                    moves += self._bishop_moves(r, c, side)
                elif piece_type == PIECE_WN:
                    moves += self._knight_moves(r, c, side)
                elif piece_type == PIECE_WP:
                    moves += self._pawn_moves(r, c, side)
        moves += self._castle_moves(side)
        return moves

    def _inside(self, r, c):
        return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS

    def _add_slide_moves(self, r, c, side, directions):
        out = []
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            while self._inside(rr, cc):
                if self.board[rr, cc] == 0:
                    out.append({"from": (r, c), "to": (rr, cc), "castle": False,
                                "promotion": False, "promo_type": None})
                else:
                    if np.sign(self.board[rr, cc]) == -side:
                        out.append({"from": (r, c), "to": (rr, cc), "castle": False,
                                    "promotion": False, "promo_type": None})
                    break
                rr += dr
                cc += dc
        return out

    def _rook_moves(self, r, c, side):
        return self._add_slide_moves(r, c, side, [(1,0),(-1,0),(0,1),(0,-1)])

    def _bishop_moves(self, r, c, side):
        return self._add_slide_moves(r, c, side, [(1,1),(1,-1),(-1,1),(-1,-1)])

    def _knight_moves(self, r, c, side):
        out = []
        for dr, dc in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
            rr, cc = r + dr, c + dc
            if self._inside(rr, cc):
                t = self.board[rr, cc]
                if t == 0 or np.sign(t) == -side:
                    out.append({"from": (r,c), "to": (rr,cc), "castle": False,
                                "promotion": False, "promo_type": None})
        return out

    def _king_moves(self, r, c, side):
        out = []
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr == dc == 0: continue
                rr, cc = r+dr, c+dc
                if not self._inside(rr, cc): continue
                t = self.board[rr, cc]
                if t == 0 or np.sign(t) == -side:
                    out.append({"from": (r,c), "to": (rr,cc), "castle": False,
                                "promotion": False, "promo_type": None})
        return out

    def _pawn_moves(self, r, c, side):
        out = []
        step = -1 if side == WHITE else 1
        fwd_r = r + step

        def add_move(rr, cc, is_capture):
            if not self._inside(rr, cc): return
            t = self.board[rr, cc]
            if is_capture and (t == 0 or np.sign(t) != -side): return
            if not is_capture and t != 0: return
            promo = self._is_promo_rank(rr, side)
            promo_types = [0,1,2] if promo else [None]
            for pt in promo_types:
                out.append({"from": (r,c), "to": (rr,cc), "castle": False,
                            "promotion": promo, "promo_type": pt})

        add_move(fwd_r, c, False)
        for dc in [-1, 1]:
            add_move(fwd_r, c+dc, True)
        return out

    def _is_promo_rank(self, r, side):
        return (r == 0 if side == WHITE else r == BOARD_ROWS - 1)

    def _castle_moves(self, side):
        out = []
        if side == WHITE:
            king_start, rook_start, king_end, rook_end = (4,3), (4,0), (4,1), (4,2)
        else:
            king_start, rook_start, king_end, rook_end = (0,0), (0,3), (0,2), (0,1)

        kr, kc = king_start
        rr, rc = rook_start
        if not self._inside(kr,kc) or not self._inside(rr,rc): return out
        if np.sign(self.board[kr,kc]) != side or abs(self.board[kr,kc]) != 1: return out
        if np.sign(self.board[rr,rc]) != side or abs(self.board[rr,rc]) != 2: return out
        if self.has_moved_king[side] or self.has_moved_rook[side]: return out
        if self._is_square_attacked(kr, kc, -side) or self._is_square_attacked(king_end[0], king_end[1], -side):
            return out
        out.append({"from": king_start, "to": king_end, "castle": True,
                    "promotion": False, "promo_type": None})
        return out

    def _is_square_attacked(self, r, c, by_side):
        for m in self._generate_pseudo_legal_moves(by_side):
            if m["to"] == (r,c): return True
        return False

    def _is_in_check(self, side):
        king_pos = None
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if self.board[r,c] != 0 and np.sign(self.board[r,c]) == side and abs(self.board[r,c]) == 1:
                    king_pos = (r,c)
                    break
            if king_pos: break
        if not king_pos: return True
        return self._is_square_attacked(*king_pos, -side)

    def _apply_move(self, m):
        fr, fc = m["from"]
        tr, tc = m["to"]
        piece = self.board[fr,fc]
        self.board[tr,tc] = piece
        self.board[fr,fc] = 0

        if m["castle"]:
            if self.side_to_move == WHITE:
                self.board[4,2] = PIECE_WR
                self.board[4,0] = 0
                self.has_moved_king[WHITE] = self.has_moved_rook[WHITE] = True
            else:
                self.board[0,1] = PIECE_BR
                self.board[0,3] = 0
                self.has_moved_king[BLACK] = self.has_moved_rook[BLACK] = True

        if m["promotion"]:
            if self.side_to_move == WHITE:
                if m["promo_type"] == 0:
                    self.board[tr,tc] = PIECE_WR
                elif m["promo_type"] == 1:
                    self.board[tr,tc] = PIECE_WN
                elif m["promo_type"] == 2:
                    self.board[tr,tc] = PIECE_WB
            else:
                if m["promo_type"] == 0:
                    self.board[tr,tc] = PIECE_BR
                elif m["promo_type"] == 1:
                    self.board[tr,tc] = PIECE_BN
                elif m["promo_type"] == 2:
                    self.board[tr,tc] = PIECE_BB

        if abs(piece) == PIECE_WK:
            self.has_moved_king[self.side_to_move] = True
        if abs(piece) == PIECE_WR:
            self.has_moved_rook[self.side_to_move] = True

        self.side_to_move *= -1
        self.move_history.append(m)

    def _check_terminal(self):
        stm = self.side_to_move
        legal = self.generate_legal_moves(stm)
        if len(legal) > 0:
            return False, None
        if self._is_in_check(stm):
            return True, 1 if -stm == WHITE else -1
        return True, 0