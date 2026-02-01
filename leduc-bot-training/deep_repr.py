# deep_repr.py

from typing import List
from env_leduc import LeducState, card_rank
import numpy as np
# Global canonical action indices for the networks
# 0 -> CHECK/CALL, 1 -> BET/RAISE, 2 -> FOLD
ACTION_INDEX = {
    "check": 0,
    "call": 0,
    "bet": 1,
    "raise": 1,
    "fold": 2,
}


def action_to_index(action: str) -> int:
    try:
        return ACTION_INDEX[action]
    except KeyError:
        raise ValueError(f"Unknown action: {action}")
    

def legal_action_mask(actions: List[str]) -> List[float]:
    """
    Given env actions like ["check", "bet"] or ["call", "fold"],
    return a mask over the 3 canonical action indices:
      mask[i] = 1 if that abstract action is available, 0 otherwise.
    """
    mask = [0.0, 0.0, 0.0]
    for a in actions:
        idx = action_to_index(a)
        mask[idx] = 1.0
    return mask
def encode_infoset(state: LeducState, player: int) -> np.ndarray:
    """
    Encode (state, player) into a fixed-length feature vector.
    Features:
      - private rank (3 one-hot)
      - public card (4 one-hot: none/J/Q/K)
      - round (2 one-hot)
      - seat (2 one-hot: am I P0 or P1)
      - is it my turn? (1)
      - pot size normalized (1)
      - current bet normalized (1)
      - raises this round bucketed (3 one-hot: 0,1,>=2)
      - last action type this round (4 one-hot: none, check/call, bet/raise, fold)
    """
    features = []

    # 1) Private card rank (3 one-hot: J,Q,K)
    priv_rank = card_rank(state.private_cards[player])  # 0,1,2
    priv_oh = [0.0, 0.0, 0.0]
    priv_oh[priv_rank] = 1.0
    features.extend(priv_oh)

    # 2) Public card (4 one-hot: none/J/Q/K)
    if state.public_card is None:
        pub_oh = [1.0, 0.0, 0.0, 0.0]  # none
    else:
        pub_oh = [0.0, 0.0, 0.0, 0.0]
        pub_rank = card_rank(state.public_card)  # 0,1,2
        pub_oh[pub_rank + 1] = 1.0  # indices 1,2,3 for J,Q,K
    features.extend(pub_oh)

    # 3) Round (2 one-hot: preflop, flop)
    rnd = state.round_index  # 0 or 1
    round_oh = [0.0, 0.0]
    if rnd in (0, 1):
        round_oh[rnd] = 1.0
    features.extend(round_oh)

    # 4) Seat (2 one-hot: P0 or P1)
    seat_oh = [0.0, 0.0]
    if player == 0:
        seat_oh[0] = 1.0
    else:
        seat_oh[1] = 1.0
    features.extend(seat_oh)

    # 5) Is it my turn? (1)
    my_turn = 1.0 if state.current_player == player else 0.0
    features.append(my_turn)

    # 6) Pot size normalized (rough scale)
    pot_norm = state.pot / 20.0  # heuristic scale
    features.append(pot_norm)

    # 7) Current bet normalized
    current_bet_norm = state.current_bet / 4.0  # bets are 2 or 4 in this game
    features.append(current_bet_norm)

    # 8) Raises this round (3 one-hot)
    this_round = state.round_index
    raises_count = 0
    last_action_type = 0  # 0: none, 1: check/call, 2: bet/raise, 3: fold

    # Extract current round history
    round_history = [a for (r, p, a) in state.history if r == this_round]
    if round_history:
        last_a = round_history[-1]
        if last_a in ("check", "call"):
            last_action_type = 1
        elif last_a in ("bet", "raise"):
            last_action_type = 2
        elif last_a == "fold":
            last_action_type = 3

    for a in round_history:
        if a in ("bet", "raise"):
            raises_count += 1

    if raises_count >= 2:
        raises_bucket = 2
    else:
        raises_bucket = raises_count  # 0,1

    raises_oh = [0.0, 0.0, 0.0]
    raises_oh[raises_bucket] = 1.0
    features.extend(raises_oh)

    # 9) Last action type this round (4 one-hot: none, check/call, bet/raise, fold)
    last_act_oh = [0.0, 0.0, 0.0, 0.0]
    last_act_oh[last_action_type] = 1.0
    features.extend(last_act_oh)

    return np.array(features, dtype=np.float32)
