# env_leduc.py
"""
Leduc Hold'em environment for CFR / Deep CFR.

Leduc Hold-em is a simplified variant of Hold'em Poker often used in ML research
Game rules (2-player Leduc):
- Deck: 6 cards = {J, Q, K} x {two suits}. 
- Suits only exist for the sake of uniqueness and do not effect hand strength.
- Players: 2. Denote them 0 and 1.
- Antes: each player antes 1 chip; Pot starts at 2.
- Cards:
    Preflop: each player gets 1 private card.
    Flop: one public card is revealed
- Betting:
    Two betting rounds: preflop (round 0) and flop (round 1).
    Betting limit:
        Preflop bet size = 2 chips.
        Flop bet size = 4 chips.
    At most MAX_RAISES_PER_ROUND raises per round. Note the initial bet counts as a raise.
    Actions:
        No bet yet this round: ["check", "bet"].
        Facing a bet and raises still allowed: ["call", "raise", "fold"].
        Facing a bet and raises capped: ["call", "fold"].
- Showdown:
    1. Two pair beats no pair.
    2. If both have a two pair or both do not, highest private ranks wins.
    3. If equal ranks and both same type, it's a tie and pot is split.

- Utilities:
    Utility for player 0 = (chips won from pot) - (their total contribution).
    Player 1's utility is the negative of player 0's.
"""
import random
from copy import deepcopy

# Card utilities

RANKS = ["J", "Q", "K"]
SUITS = ["a", "b"]  # two arbitrarily labelled suits
DECK = [r + s for r in RANKS for s in SUITS]  # ["Ja", "Qa", "Ka", "Jb", "Qb", "Kb"] at start of game

RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}  # J=0, Q=1, K=2

def card_rank(card: str) -> int:
    # returns card rank
    return RANK_TO_INT[card[0]]

# Betting constants

PRE_FLOP = 0
FLOP = 1
BET_SIZES = {
    PRE_FLOP: 2,
    FLOP: 4,
}
MAX_RAISES_PER_ROUND = 2 
CHECK = "check"
BET = "bet"
CALL = "call"
RAISE = "raise"
FOLD = "fold"

# Game state

class LeducState:
    """
    Represents a single state of a Leduc game as defined above.
    To be used by CFR / Deep CFR.
    Fields:
        deck: list of remaining cards
        private_cards: [card_p0, card_p1]
        public_card: None (preflop) or string card (flop)
        current_player: 0 or 1
        round_index: 0=preflop, 1=flop
        total_bets: [chips contributed by p0, chips contributed by p1]
        round_bets: contributions this round 
        current_bet: highest contribution in this round 
        num_raises: number of bets/raises so far this round
        prev_action: last non-terminal action in this round ("check" or "bet" or None)
        terminal: bool
        winner: None if not decided yet or showdown; 0/1 if fold-based win
        history: list of (round_index, player, action) for debugging / info
    """

    def __init__(
        self,
        deck,
        private_cards,
        public_card,
        current_player,
        round_index,
        total_bets,
        round_bets,
        current_bet,
        num_raises,
        prev_action,
        terminal=False,
        winner=None,
        history=None,
    ):
        self.deck = deck
        self.private_cards = private_cards
        self.public_card = public_card
        self.current_player = current_player
        self.round_index = round_index
        self.total_bets = total_bets
        self.round_bets = round_bets
        self.current_bet = current_bet
        self.num_raises = num_raises
        self.prev_action = prev_action
        self.terminal = terminal
        self.winner = winner
        self.history = history or []

    @property
    def pot(self) -> int:
        # Total pot = \sigma contributions
        return self.total_bets[0] + self.total_bets[1]

    def clone(self):
        #Deep copy of the state (for CFR type traversal).
        return LeducState(
            deck=list(self.deck),
            private_cards=list(self.private_cards),
            public_card=self.public_card,
            current_player=self.current_player,
            round_index=self.round_index,
            total_bets=list(self.total_bets),
            round_bets=list(self.round_bets),
            current_bet=self.current_bet,
            num_raises=self.num_raises,
            prev_action=self.prev_action,
            terminal=self.terminal,
            winner=self.winner,
            history=list(self.history),
        )
    def is_terminal(self) -> bool:
        return self.terminal

    # Legal actions

    def legal_actions(self):
        #Returns a list of legal actions for the curret state.
        if self.terminal:
            return []
        actions = []
        # No bet yet in this round
        if self.current_bet == 0:
            # P0 and P1 have matched contributions for the round round
            # Actions: check or bet (if max raises not yet reached)
            actions.append(CHECK)
            if self.num_raises < MAX_RAISES_PER_ROUND:
                actions.append(BET)
        else:
            # Outstanding bet facing current P.
            if self.num_raises < MAX_RAISES_PER_ROUND:
                actions.extend([CALL, RAISE, FOLD])
            else:
                actions.extend([CALL, FOLD])
        return actions

    # State transitions

    def next_state(self, action: str):
        # Return the next state after action taken.
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action} in state with actions {self.legal_actions()}")
        s = self.clone()
        p = s.current_player
        opp = 1 - p
        s.history.append((s.round_index, p, action))
        if action == CHECK:
            if s.current_bet != 0:
                raise RuntimeError("Cannot CHECK when there is a bet to call.")

            if s.prev_action == CHECK:
                # second check in a row so betting ends
                if s.round_index == FLOP:
                    # end of game so go to showdown
                    s.terminal = True
                    s.winner = None  # showdown
                else:
                    # flop
                    s._advance_round()
            else:
                # First check this round
                s.prev_action = CHECK
                s.current_player = opp
        elif action == BET:
            if s.current_bet != 0:
                raise RuntimeError("Cannot BET when a bet already exists.")
            if s.num_raises >= MAX_RAISES_PER_ROUND:
                raise RuntimeError("No raises left this round.")

            bet_size = BET_SIZES[s.round_index]
            s.round_bets[p] += bet_size
            s.total_bets[p] += bet_size
            s.current_bet = s.round_bets[p]
            s.num_raises += 1
            s.prev_action = BET
            s.current_player = opp

        elif action == CALL:
            if s.current_bet == 0:
                raise RuntimeError("Nothing to CALL.")

            to_call = s.current_bet - s.round_bets[p]
            if to_call < 0:
                raise RuntimeError("Player already matched or exceeded bet; cannot CALL.")

            s.round_bets[p] += to_call
            s.total_bets[p] += to_call

            # After a call, betting round end.
            if s.round_index == FLOP:
                # showdown
                s.terminal = True
                s.winner = None
            else:
                s._advance_round()
        elif action == RAISE:
            if s.current_bet == 0:
                raise RuntimeError("Cannot RAISE when no bet exists.")
            if s.num_raises >= MAX_RAISES_PER_ROUND:
                raise RuntimeError("Raise cap reached.")

            bet_size = BET_SIZES[s.round_index]
            new_bet = s.current_bet + bet_size
            to_put_in = new_bet - s.round_bets[p]

            s.round_bets[p] += to_put_in
            s.total_bets[p] += to_put_in
            s.current_bet = new_bet
            s.num_raises += 1
            s.prev_action = BET  # last action was a bet/raise
            s.current_player = opp

        elif action == FOLD:
            if s.current_bet == 0:
                raise RuntimeError("Cannot FOLD when no bet exists.")
            s.terminal = True
            s.winner = opp  # opponent wins pot

        else:
            raise ValueError("Unknown action: " + action)

        return s

    def _advance_round(self):
        #preflop to flop or showdown if on flop.
        if self.round_index == PRE_FLOP:
            # Reveal public card
            if self.public_card is not None:
                raise RuntimeError("Public card already set.")
            if not self.deck:
                raise RuntimeError("Deck empty; cannot deal public card.")
            self.public_card = self.deck.pop()  # deal public card
            self.round_index = FLOP
            # Reset round betting
            self.round_bets = [0, 0]
            self.current_bet = 0
            self.num_raises = 0
            self.prev_action = None
            self.current_player = 0  # player 0 goes first each round (we seat-average our results to account for this)
        else:
            #Already on flop so showdown
            self.terminal = True
            self.winner = None

    # Utility/showdown
    def showdown_winner(self):
        # Determines winner at showdown
        # 0 if P0 wins, 1 if P1 wins, else tie
        if self.public_card is None:
            raise RuntimeError("Showdown called without public card.")

        board_rank = card_rank(self.public_card)
        r0 = card_rank(self.private_cards[0])
        r1 = card_rank(self.private_cards[1])

        p0_pair = (r0 == board_rank)
        p1_pair = (r1 == board_rank)

        if p0_pair and not p1_pair:
            return 0
        elif p1_pair and not p0_pair:
            return 1
        else:
            # compare high card
            if r0 > r1:
                return 0
            elif r1 > r0:
                return 1
            else:
                return None

    def utility(self, player: int) -> float:
        """
        Return terminal utility for player
        Positive = chips won.
        Only in terminal states.
        """
        if not self.terminal:
            raise RuntimeError("Utility called on non-terminal state.")

        pot = self.pot
        c0, c1 = self.total_bets

        # Fold outcome
        if self.reason_is_fold:
            if self.winner == 0:
                u0 = pot - c0
            else:
                u0 = -c0
        else:
            # Showdown
            sw = self.showdown_winner()
            if sw == 0:
                u0 = pot - c0
            elif sw == 1:
                u0 = -c0
            else:
                # Tie -> split pot
                u0 = pot / 2.0 - c0

        return u0 if player == 0 else -u0

    @property
    def reason_is_fold(self) -> bool:
        return self.terminal and self.winner is not None


# Environment helper

class LeducEnv:
    #Simple helper to create new random Leduc games.

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def new_game(self) -> LeducState:
        """
        Create a new random Leduc game:
            - Shuffle deck
            - Deal one private card to each player
            - Take antes
            - Start preflop betting with player 0 to act
        """
        deck = DECK[:]
        self.rng.shuffle(deck)

        total_bets = [1, 1]

        # Deal private cards
        private0 = deck.pop()
        private1 = deck.pop()
        private_cards = [private0, private1]

        state = LeducState(
            deck=deck,
            private_cards=private_cards,
            public_card=None,
            current_player=0,
            round_index=PRE_FLOP,
            total_bets=total_bets,
            round_bets=[0, 0],
            current_bet=0,
            num_raises=0,
            prev_action=None,
            terminal=False,
            winner=None,
            history=[],
        )
        return state


# Test

if __name__ == "__main__":
    env = LeducEnv(seed=42)
    s = env.new_game()
    print("Initial state:")
    print(f"  P0 card: {s.private_cards[0]}")
    print(f"  P1 card: {s.private_cards[1]}")
    print(f"  Pot: {s.pot}")
    print(f"  Legal actions for P{s.current_player}: {s.legal_actions()}")

    # Example Random playout
    while not s.is_terminal():
        acts = s.legal_actions()
        a = random.choice(acts)
        print(f"P{s.current_player} does {a}")
        s = s.next_state(a)
        print(f"  Pot now: {s.pot}, round: {s.round_index}, public: {s.public_card}")
        print(f"  Round bets: {s.round_bets}, total bets: {s.total_bets}")
        if s.is_terminal():
            break

    print("\nTerminal state reached.")
    if s.reason_is_fold:
        print(f"Ended by fold. Winner: P{s.winner}")
    else:
        print(f"Showdown! Public card: {s.public_card}")
        print(f"P0 card: {s.private_cards[0]}, P1 card: {s.private_cards[1]}")
        print(f"Showdown winner: {s.showdown_winner()}")
    print(f"Utility P0: {s.utility(0)}, Utility P1: {s.utility(1)}")
