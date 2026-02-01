from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CreateGameRequest(BaseModel):
    human_player: int = Field(..., ge=0, le=1)


class ActionRequest(BaseModel):
    action: str


class PublicState(BaseModel):
    round_index: int
    pot: int
    current_bet: int
    num_raises: int
    total_bets: List[int]
    current_player: int
    terminal: bool
    public_card: Optional[str]
    private_card: str
    history: List[List[Any]]
    legal_actions: List[str]


class ActionResult(BaseModel):
    actor: int
    action: str
    confidence: float
    policy_probs: Dict[str, float]


class CreateGameResponse(BaseModel):
    game_id: str
    hand_id: str
    state: PublicState


class ActResponse(BaseModel):
    game_id: str
    hand_id: str
    state: PublicState
    human_action: Optional[ActionResult] = None
    bot_action: Optional[ActionResult] = None
    terminal: bool
    winner: Optional[int] = None
    utility_p0: Optional[float] = None
    utility_p1: Optional[float] = None


class NewHandResponse(BaseModel):
    game_id: str
    hand_id: str
    state: PublicState


class ActionLog(BaseModel):
    action_index: int
    actor: int
    action: str
    confidence: Optional[float]
    policy_probs: Optional[Dict[str, float]]
    round_index: int
    pot: int
    current_bet: int
    num_raises: int
    created_at: str


class ActionsResponse(BaseModel):
    game_id: str
    hand_id: str
    actions: List[ActionLog]


class StatsResponse(BaseModel):
    total_games: int
    total_hands: int
    total_actions: int
    bot_action_freq: Dict[str, int]
    human_action_freq: Dict[str, int]
    avg_bot_confidence: Optional[float]


class HealthResponse(BaseModel):
    status: str
