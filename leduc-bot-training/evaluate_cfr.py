# evaluate_cfr.py
"""
Evaluate the tabular CFR strategy for Leduc Hold'em.
1. Train tabular CFR (or assume it's already trained in this process).
2. Define policy functions for:
    - CFR policy (using infosets + average strategies)
    - Random policy
    - Simple heuristic policy
3. Simulate many hands and report average payoff for Player 0.
"""
import random
from typing import Callable, List

from env_leduc import LeducEnv, LeducState, card_rank
from cfr_tabular import get_average_strategy_for_key, info_key, train_cfr


# -------------------
# Policy functions
# -------------------

def cfr_policy(state: LeducState, player: int, actions: List[str]) -> List[float]:
    """
    Policy for the tabular CFR agent: use the average strategy from the InfoSet,
    fall back to uniform if unseen.
    """
    key = info_key(state, player, actions)
    avg = get_average_strategy_for_key(key, actions)
    # avg length should match actions length; safety normalize
    s = sum(avg)
    if s <= 0:
        n = len(actions)
        return [1.0 / n] * n
    return [x / s for x in avg]


def random_policy(state: LeducState, player: int, actions: List[str]) -> List[float]:
    """
    Baseline: choose uniformly among legal actions.
    """
    n = len(actions)
    return [1.0 / n] * n

def simple_heuristic_policy(state: LeducState, player: int, actions: List[str]) -> List[float]:
    """
    Very rough human-ish baseline policy:
      - Preflop:
          If no bet yet:
            - Bet with K always, Q half the time, J never.
          Facing a bet:
            - Call with K, fold J, mix with Q.
      - Flop:
          If you have a pair with the board: always continue (bet/call).
          Otherwise: mostly check/fold.
    """
    # Default uniform fallback
    n = len(actions)
    probs = [1.0 / n] * n

    # Basic info
    priv_rank = card_rank(state.private_cards[player])
    board_rank = card_rank(state.public_card) if state.public_card is not None else None
    no_bet_yet = (state.current_bet == 0)

    # Helper to map an action name to index if present
    def idx(act: str):
        try:
            return actions.index(act)
        except ValueError:
            return None

    # Preflop logic
    if state.round_index == 0:
        if no_bet_yet:
            i_check = idx("check")
            i_bet = idx("bet")
            if i_check is not None and i_bet is not None:
                probs = [0.0] * n
                if priv_rank == 2:   # K
                    # Always bet
                    probs[i_bet] = 1.0
                elif priv_rank == 1: # Q
                    # 50/50 check/bet
                    probs[i_check] = 0.5
                    probs[i_bet] = 0.5
                else:                # J
                    # Always check
                    probs[i_check] = 1.0
        else:
            # Facing a bet
            i_call = idx("call")
            i_raise = idx("raise")
            i_fold = idx("fold")
            if i_call is not None and i_fold is not None:
                probs = [0.0] * n
                if priv_rank == 2:   # K
                    # Call (and sometimes raise if available)
                    if i_raise is not None:
                        probs[i_raise] = 0.2
                        probs[i_call] = 0.8
                    else:
                        probs[i_call] = 1.0
                elif priv_rank == 1: # Q
                    # Mix call and fold
                    probs[i_call] = 0.5
                    probs[i_fold] = 0.5
                else:                # J
                    # Mostly fold
                    probs[i_fold] = 0.9
                    probs[i_call] = 0.1

    # Flop logic
    else:
        has_pair = (board_rank is not None and priv_rank == board_rank)
        i_check = idx("check")
        i_bet = idx("bet")
        i_call = idx("call")
        i_raise = idx("raise")
        i_fold = idx("fold")

        if no_bet_yet and i_check is not None and i_bet is not None:
            probs = [0.0] * n
            if has_pair:
                # Value bet
                probs[i_bet] = 1.0
            else:
                # Mostly check weak hands
                probs[i_check] = 1.0
        elif not no_bet_yet and i_call is not None and i_fold is not None:
            probs = [0.0] * n
            if has_pair:
                # Always continue (call, occasionally raise)
                if i_raise is not None:
                    probs[i_raise] = 0.2
                    probs[i_call] = 0.8
                else:
                    probs[i_call] = 1.0
            else:
                # Mostly fold when weak
                probs[i_fold] = 0.9
                probs[i_call] = 0.1
    # Normalize safety
    s = sum(probs)
    if s <= 0:
        return [1.0 / n] * n
    return [p / s for p in probs]


# Simulation helpers

def sample_action(actions: List[str], probs: List[float], rng: random.Random) -> str:
    # Sample an action index according to probs.
    # Safety normalize
    total = sum(probs)
    if total <= 0:
        n = len(actions)
        probs = [1.0 / n] * n
        total = 1.0
    cum = 0.0
    r = rng.random() * total
    for a, p in zip(actions, probs):
        cum += p
        if r <= cum:
            return a
    return actions[-1]


def play_hand(env: LeducEnv,
              policy_p0: Callable[[LeducState, int, List[str]], List[float]],
              policy_p1: Callable[[LeducState, int, List[str]], List[float]],
              rng: random.Random) -> float:
    # Play a single hand of Leduc between two policies.
    # Returns Player 0's utility (chips won).
    state = env.new_game()
    while not state.is_terminal():
        player = state.current_player
        actions = state.legal_actions()
        if player == 0:
            probs = policy_p0(state, player, actions)
        else:
            probs = policy_p1(state, player, actions)
        action = sample_action(actions, probs, rng)
        state = state.next_state(action)
    return state.utility(0)


def evaluate_match(policy_p0, policy_p1, num_hands=100000, seed=123) -> float:
    # Evaluate average payoff for P0 over many hands.
    env = LeducEnv(seed=seed)
    rng = random.Random(seed + 1)
    total = 0.0
    for _ in range(num_hands):
        total += play_hand(env, policy_p0, policy_p1, rng)
    return total / num_hands


# Main: train + evaluate

if __name__ == "__main__":
    # 1) Train tabular CFR
    print("Training tabular CFR policy...")
    train_cfr(iterations=200000)

    print("Evaluating CFR vs random...")
    ev_cfr_vs_random = evaluate_match(cfr_policy, random_policy, num_hands=50000)
    print(f"EV (P0) CFR vs random: {ev_cfr_vs_random:.4f} chips/hand")

    print("Evaluating CFR vs heuristic...")
    ev_cfr_vs_heur = evaluate_match(cfr_policy, simple_heuristic_policy, num_hands=50000)
    print(f"EV (P0) CFR vs heuristic: {ev_cfr_vs_heur:.4f} chips/hand")

    print("Evaluating heuristic vs random (sanity check)...")
    ev_heur_vs_random = evaluate_match(simple_heuristic_policy, random_policy, num_hands=50000)
    print(f"EV (P0) heuristic vs random: {ev_heur_vs_random:.4f} chips/hand")

    # heuristic as P0, CFR as P1
    ev_heur_as_p0_vs_cfr_as_p1 = evaluate_match(simple_heuristic_policy, cfr_policy, num_hands=50000)
    print("EV (P0) heuristic vs CFR:", ev_heur_as_p0_vs_cfr_as_p1)
