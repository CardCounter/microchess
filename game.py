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
PROMO_ROOK = 0
PROMO_KNIGHT = 1
PROMO_BISHOP = 2


class MicrochessEnv:
    def __init__(self):
        self.board = None
        self.side_to_move = WHITE
        self.has_moved_king = {WHITE: False, BLACK: False}
        self.has_moved_rook = {WHITE: False, BLACK: False}
        self.move_history = []
        self.halfmove_clock = 0  # counts plies since last capture or pawn move
        self.repetition_history = []  # list of board hash strings for repetition tracking
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
        self.halfmove_clock = 0
        self.repetition_history = [self._board_hash()]
        return self._get_obs()

    def _get_obs(self):
        flat_board = self.board.flatten().astype(np.int8)
        stm = np.int8(self.side_to_move)
        return np.concatenate([flat_board, np.array([stm], dtype=np.int8)], axis=0)

    def _board_hash(self):
        # simple hash: board layout + side to move + castle rights
        # (castle rights matter because they affect legality of future moves)
        parts = [str(int(x)) for x in self.board.flatten()]
        parts.append(str(int(self.side_to_move)))
        parts.append("K" if not self.has_moved_king[WHITE] else "k")
        parts.append("R" if not self.has_moved_rook[WHITE] else "r")
        parts.append("K" if not self.has_moved_king[BLACK] else "k")
        parts.append("R" if not self.has_moved_rook[BLACK] else "r")
        return ",".join(parts)

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
                    if promo_bucket == PROMO_ROOK: # bucket 0
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

    def _generate_pseudo_legal_moves(self, side, include_castle=True):
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
        if include_castle:
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
            if not self._inside(rr, cc):
                return
            t = self.board[rr, cc]
            # capture move: square must contain enemy piece
            if is_capture:
                if t == 0 or np.sign(t) != -side:
                    return
            else:
                # quiet move: square must be empty
                if t != 0:
                    return

            promo = self._is_promo_rank(rr, side)
            promo_types = [PROMO_ROOK, PROMO_KNIGHT, PROMO_BISHOP] if promo else [None]
            for pt in promo_types:
                out.append({
                    "from": (r, c),
                    "to": (rr, cc),
                    "castle": False,
                    "promotion": promo,
                    "promo_type": pt
                })

        # single step forward
        add_move(fwd_r, c, False)

        # double step forward (only from starting rank, path must be clear, and landing square clear)
        if side == WHITE:
            start_rank = BOARD_ROWS - 2  # row 3 in 0-indexed 5x4
        else:
            start_rank = 1               # row 1 for black in initial position

        two_r = r + 2 * step
        if r == start_rank:
            # need inside board, intermediate empty, landing empty
            if self._inside(fwd_r, c) and self.board[fwd_r, c] == 0:
                if self._inside(two_r, c) and self.board[two_r, c] == 0:
                    # can't be a promotion because you're skipping over promo rank anyway,
                    # but we'll still run through add_move for consistency
                    add_move(two_r, c, False)

        # diagonal captures
        for dc in [-1, 1]:
            add_move(fwd_r, c + dc, True)

        return out

    def _is_promo_rank(self, r, side):
        return (r == 0 if side == WHITE else r == BOARD_ROWS - 1)

    def _castle_moves(self, side):
        out = []
        if side == WHITE:
            king_start, rook_start, king_end = (4,3), (4,0), (4,1)
        else:
            king_start, rook_start, king_end = (0,0), (0,3), (0,2)

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
        for m in self._generate_pseudo_legal_moves(by_side, include_castle=False):
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
        target = self.board[tr,tc]

        # detect capture or pawn move for 50-move rule reset
        is_capture = (target != 0)
        is_pawn_move = (abs(piece) == PIECE_WP)

        # move piece
        self.board[tr,tc] = piece
        self.board[fr,fc] = 0

        # handle castling rook movement
        if m["castle"]:
            if self.side_to_move == WHITE:
                # white king goes (4,3)->(4,1), rook (4,0)->(4,2) per rules:
                # NOTE: in _castle_moves we defined king_end as (4,1) for white and (0,2) for black,
                # and we later hard-move rook squares to match that convention.
                # Here we mirror the logic used earlier in your code.
                self.board[4,2] = PIECE_WR
                self.board[4,0] = 0
                self.has_moved_king[WHITE] = True
                self.has_moved_rook[WHITE] = True
            else:
                self.board[0,1] = PIECE_BR
                self.board[0,3] = 0
                self.has_moved_king[BLACK] = True
                self.has_moved_rook[BLACK] = True

        # handle promotion
        if m["promotion"]:
            if self.side_to_move == WHITE:
                if m["promo_type"] == PROMO_ROOK:
                    self.board[tr,tc] = PIECE_WR
                elif m["promo_type"] == 1:
                    self.board[tr,tc] = PIECE_WN
                elif m["promo_type"] == 2:
                    self.board[tr,tc] = PIECE_WB
            else:
                if m["promo_type"] == PROMO_ROOK:
                    self.board[tr,tc] = PIECE_BR
                elif m["promo_type"] == 1:
                    self.board[tr,tc] = PIECE_BN
                elif m["promo_type"] == 2:
                    self.board[tr,tc] = PIECE_BB

        # update moved flags for rook/king
        if abs(piece) == PIECE_WK:
            self.has_moved_king[self.side_to_move] = True
        if abs(piece) == PIECE_WR:
            self.has_moved_rook[self.side_to_move] = True

        # update 50-move (really 100-ply) clock
        if is_capture or is_pawn_move or m["promotion"]:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # flip side and record position / history
        self.side_to_move *= -1
        self.move_history.append(m)
        self.repetition_history.append(self._board_hash())

    def _is_insufficient_material(self):
        """
        Return True if it's impossible to force mate.
        We consider these positions auto-draw:
          - K vs K
          - K+N vs K
          - K+B vs K
        We do NOT auto-draw K+R vs K or K+P vs K because rook can mate
        and pawn can promote on this board.
        """
        pieces = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.board[r, c]
                if p != 0:
                    pieces.append(p)

        # K vs K
        if len(pieces) == 2 and all(abs(x) == PIECE_WK for x in pieces):
            return True

        # exactly 3 pieces: K + minor vs K
        if len(pieces) == 3:
            absvals = [abs(x) for x in pieces]
            # need exactly two kings
            if absvals.count(PIECE_WK) == 2:
                # find the non-king
                for x in pieces:
                    ax = abs(x)
                    if ax != PIECE_WK:
                        if ax == PIECE_WN or ax == PIECE_WB:
                            return True
        return False

    def _is_fifty_move_draw(self):
        """
        50-move rule analogue:
        In chess it's 50 moves (i.e. 100 plies) with no pawn move or capture.
        We're tracking half-moves (plies). If halfmove_clock >= 100 -> draw.
        """
        return self.halfmove_clock >= 100

    def _is_threefold_repetition(self):
        """
        Threefold repetition: same position (including side to move and castle rights)
        has appeared at least 3 times.
        """
        counts = {}
        for h in self.repetition_history:
            counts[h] = counts.get(h, 0) + 1
            if counts[h] >= 3:
                return True
        return False

    def _check_terminal(self):
        stm = self.side_to_move
        legal = self.generate_legal_moves(stm)

        # no legal moves -> checkmate or stalemate
        if len(legal) == 0:
            if self._is_in_check(stm):
                # checkmate: side to move is mated, other side wins
                return True, 1 if -stm == WHITE else -1
            else:
                # stalemate draw
                return True, 0

        # insufficient material draw
        if self._is_insufficient_material():
            return True, 0

        # 50-move rule draw
        if self._is_fifty_move_draw():
            return True, 0

        # threefold repetition draw
        if self._is_threefold_repetition():
            return True, 0

        # otherwise game continues
        return False, None


__all__ = ["MicrochessEnv"]

if __name__ == "__main__":
    env = MicrochessEnv()
    obs = env.reset()
    print("Observation shape:", obs.shape)
    print("Number of actions:", ACTION_SIZE)