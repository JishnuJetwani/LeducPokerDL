# cfr_tabular.py
"""
Tabular CFR for 2-player Leduc Hold'em using env_leduc.py.

This is a baseline solver:
- Uses exact information sets (no function approximation).
- Runs chance-sampled CFR: each iteration samples a random deal.
- Updates regrets/strategies over time.

We will later replace the tabular InfoSet with neural nets (Deep CFR), but this file lets us:
  - Verify the environment is correct.
  - Get an approximate equilibrium strategy for comparison.
"""

import random
import numpy as np
from env_leduc import (
    LeducEnv,
    LeducState,
    PRE_FLOP,
    FLOP,
    CHECK, BET, CALL, RAISE, FOLD,
    card_rank,
)


# Global infoset table: key -> id
infosets = {}
_action_offsets = []
_action_sizes = []
_action_lists = []
_regret_sum = np.zeros(0, dtype=np.float64)
_strategy_sum = np.zeros(0, dtype=np.float64)


def _ensure_infoset(key, actions):
    global _regret_sum, _strategy_sum
    if key in infosets:
        infoset_id = infosets[key]
        if _action_sizes[infoset_id] != len(actions):
            raise ValueError("Infoset action size mismatch for key: " + key)
        return infoset_id
    infoset_id = len(_action_offsets)
    infosets[key] = infoset_id
    offset = _regret_sum.shape[0]
    size = len(actions)
    _action_offsets.append(offset)
    _action_sizes.append(size)
    _action_lists.append(list(actions))
    _regret_sum = np.concatenate([_regret_sum, np.zeros(size, dtype=np.float64)])
    _strategy_sum = np.concatenate([_strategy_sum, np.zeros(size, dtype=np.float64)])
    return infoset_id


def _slice_for(infoset_id):
    offset = _action_offsets[infoset_id]
    size = _action_sizes[infoset_id]
    return offset, size


def get_strategy_for_id(infoset_id, reach_prob):
    offset, size = _slice_for(infoset_id)
    regrets = _regret_sum[offset:offset + size]
    positive = np.maximum(regrets, 0.0)
    normalizing_sum = positive.sum()
    if normalizing_sum > 0:
        return positive / normalizing_sum
    return np.full(size, 1.0 / size, dtype=np.float64)


def get_average_strategy_for_id(infoset_id):
    offset, size = _slice_for(infoset_id)
    strat = _strategy_sum[offset:offset + size]
    normalizing_sum = strat.sum()
    if normalizing_sum > 0:
        return strat / normalizing_sum
    return np.full(size, 1.0 / size, dtype=np.float64)


def get_average_strategy_for_key(key, actions):
    if key not in infosets:
        n = len(actions)
        return [1.0 / n] * n
    infoset_id = infosets[key]
    return get_average_strategy_for_id(infoset_id).tolist()


# Info set key

def info_key(state: LeducState, player: int, actions):
    """
    Build an information set key from the POV of P.
    In Leduc, an info set contains:
      - player index
      - round (preflop/flop)
      - private card rank
      - public card rank or "none"
      - betting history (sequence of (round, player, action)), but without card info
    We'll compress history into a string:
      "0:c-1:b-1:r"  (round, player, first letter of action)
    """
    private_card = state.private_cards[player]
    private_rank = card_rank(private_card)  # 0,1,2 for J,Q,K

    if state.public_card is None:
        public_repr = "N"  # None
    else:
        public_repr = str(card_rank(state.public_card))  # 0/1/2

    # Encode betting history
    # We DON'T include card info there, only actions & who acted.
    hist_parts = []
    for (rnd, p, a) in state.history:
        hist_parts.append(f"{rnd}{p}{a[0]}")  # e.g. "0c" for check, "1b" for bet
    hist_str = "|".join(hist_parts) if hist_parts else "start"

    key = f"P{player}|R{state.round_index}|priv{private_rank}|pub{public_repr}|H{hist_str}"
    return key


# CFR recursion

def cfr(state: LeducState, traverser: int, p0: float, p1: float):
    """
    CFR minimization recursion.
    Returns utility for traverser from this state,
    assuming both players follow CFR strategies from infosets.
    traverser: which player (0 or 1) we are updating regrets for.
    p0, p1: reach probabilities of players 0 and 1, respectively.
    """
    if state.is_terminal():
        # Utility from the traverser's perspective
        return state.utility(traverser)

    current_player = state.current_player
    actions = state.legal_actions()

    # Build info set key for current_player
    key = info_key(state, current_player, actions)
    infoset_id = _ensure_infoset(key, actions)

    # Get strategy via regret matching
    if current_player == 0:
        strategy = get_strategy_for_id(infoset_id, p0)
    else:
        strategy = get_strategy_for_id(infoset_id, p1)
    # at this point we have actions and probability for each action
    # Recursively compute utility for each action
    utils = np.zeros(len(actions), dtype=np.float64)
    node_util = 0.0

    for i, a in enumerate(actions):
        next_state = state.next_state(a)

        if current_player == 0:
            util = cfr(next_state, traverser, p0 * strategy[i], p1)
        else:
            util = cfr(next_state, traverser, p0, p1 * strategy[i])

        utils[i] = util
        node_util += strategy[i] * util

    # Regret update only at infosets belonging to traverser
    if current_player == traverser:
        opp_reach = p1 if traverser == 0 else p0
        offset, size = _slice_for(infoset_id)
        for i in range(size):
            regret = utils[i] - node_util
            _regret_sum[offset + i] += opp_reach * regret

    # Strategy sum update (for average strategy)
    reach = p0 if current_player == 0 else p1
    offset, size = _slice_for(infoset_id)
    for i in range(size):
        _strategy_sum[offset + i] += reach * strategy[i]

    return node_util


# Training loop

def train_cfr(iterations=10000, seed=42):
    env = LeducEnv(seed=seed)
    for t in range(iterations):
        # Sample one random deal (chance sampling)
        state = env.new_game()

        # Run CFR with player 0 as traverser
        cfr(state, traverser=0, p0=1.0, p1=1.0)
        # And with player 1 as traverser
        cfr(state, traverser=1, p0=1.0, p1=1.0)

        if (t + 1) % max(1, iterations // 10) == 0:
            print(f"Iteration {t+1}/{iterations}...")

    print("Training complete.")
    print(f"Number of info sets: {len(infosets)}")


def print_sample_strategies(num=20):
    # Print a few average strategies to inspect what the CFR solution looks like.
    print("\nSample average strategies:")
    count = 0
    for key, infoset_id in infosets.items():
        avg = get_average_strategy_for_id(infoset_id).tolist()
        print(f"{key}: {avg}")
        count += 1
        if count >= num:
            break

def print_key_strategies():
    #Print strategies for a few key infosets to inspect preflop behavior:
    #  - P0 with K at start
    #  - P0 with J at start
    keys_of_interest = [
        "P0|R0|priv2|pubN|Hstart",  # P0, preflop, K, no public, start
        "P0|R0|priv1|pubN|Hstart",  # P0, preflop, Q, no public, start
        "P0|R0|priv0|pubN|Hstart",  # P0, preflop, J, no public, start
    ]
    print("\n=== Key preflop infosets (P0 at start) ===")
    for k in keys_of_interest:
        infoset_id = infosets.get(k)
        if infoset_id is None:
            print(f"{k}: (not visited)")
        else:
            avg = get_average_strategy_for_id(infoset_id).tolist()
            print(f"{k}: {avg}")

if __name__ == "__main__":
    train_cfr(iterations=50000)
    print_key_strategies()
    print_sample_strategies(num=20)
