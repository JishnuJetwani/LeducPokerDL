import sys
from pathlib import Path
from typing import Any, Dict, List

from app.config import TRAINING_DIR

# Ensure training code is importable
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

from env_leduc import LeducEnv, LeducState  # type: ignore


def serialize_state(state: LeducState) -> Dict[str, Any]:
    return {
        "deck": list(state.deck),
        "private_cards": list(state.private_cards),
        "public_card": state.public_card,
        "current_player": state.current_player,
        "round_index": state.round_index,
        "total_bets": list(state.total_bets),
        "round_bets": list(state.round_bets),
        "current_bet": state.current_bet,
        "num_raises": state.num_raises,
        "prev_action": state.prev_action,
        "terminal": state.terminal,
        "winner": state.winner,
        "history": [list(h) for h in state.history],
    }


def deserialize_state(data: Dict[str, Any]) -> LeducState:
    return LeducState(
        deck=list(data["deck"]),
        private_cards=list(data["private_cards"]),
        public_card=data.get("public_card"),
        current_player=int(data["current_player"]),
        round_index=int(data["round_index"]),
        total_bets=list(data["total_bets"]),
        round_bets=list(data["round_bets"]),
        current_bet=int(data["current_bet"]),
        num_raises=int(data["num_raises"]),
        prev_action=data.get("prev_action"),
        terminal=bool(data["terminal"]),
        winner=data.get("winner"),
        history=[tuple(h) for h in data.get("history", [])],
    )


def new_game_state(seed: int | None = None) -> LeducState:
    env = LeducEnv(seed=seed)
    return env.new_game()


def public_state_view(state: LeducState, human_player: int) -> Dict[str, Any]:
    actions = state.legal_actions()
    return {
        "round_index": state.round_index,
        "pot": state.pot,
        "current_bet": state.current_bet,
        "num_raises": state.num_raises,
        "total_bets": list(state.total_bets),
        "current_player": state.current_player,
        "terminal": state.terminal,
        "public_card": state.public_card,
        "private_card": state.private_cards[human_player],
        "history": [list(h) for h in state.history],
        "legal_actions": actions,
    }
