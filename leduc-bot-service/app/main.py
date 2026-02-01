from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session

from app.config import DEFAULT_BOT_PLAYER
from app.db import SessionLocal
from app.game import deserialize_state, new_game_state, public_state_view, serialize_state
from app.models import Action, Game, Hand
from app.policy import PolicyEngine
from app.schemas import (
    ActResponse,
    ActionRequest,
    ActionResult,
    ActionsResponse,
    ActionLog,
    CreateGameRequest,
    CreateGameResponse,
    HealthResponse,
    NewHandResponse,
    StatsResponse,
)

app = FastAPI(title="Leduc CFR Service", version="0.1.0")
policy_engine: Optional[PolicyEngine] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def load_policy():
    global policy_engine
    policy_engine = PolicyEngine()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/games", response_model=CreateGameResponse)
def create_game(payload: CreateGameRequest, db: Session = Depends(get_db)):
    human_player = payload.human_player
    bot_player = 1 - human_player if DEFAULT_BOT_PLAYER not in (0, 1) else DEFAULT_BOT_PLAYER
    if bot_player == human_player:
        bot_player = 1 - human_player

    state = new_game_state()
    game = Game(
        human_player=human_player,
        bot_player=bot_player,
        current_state=serialize_state(state),
        current_player=state.current_player,
        terminal=state.terminal,
        status="active",
    )
    hand = Hand(game=game, hand_index=0)

    db.add(game)
    db.add(hand)
    db.commit()
    db.refresh(game)
    db.refresh(hand)

    public_state = public_state_view(state, human_player)
    return CreateGameResponse(game_id=str(game.id), hand_id=str(hand.id), state=public_state)


def log_action(
    db: Session,
    game: Game,
    hand: Hand,
    action_index: int,
    actor: int,
    action: str,
    legal_actions: list,
    state_before: dict,
    state_after: dict,
    confidence: float | None,
    policy_probs: dict | None,
):
    action_row = Action(
        game_id=game.id,
        hand_id=hand.id,
        action_index=action_index,
        actor=actor,
        action=action,
        legal_actions=legal_actions,
        policy_probs=policy_probs,
        confidence=confidence,
        round_index=state_before["round_index"],
        pot=state_before["total_bets"][0] + state_before["total_bets"][1],
        current_bet=state_before["current_bet"],
        num_raises=state_before["num_raises"],
        state_before=state_before,
        state_after=state_after,
    )
    db.add(action_row)


def get_latest_hand(db: Session, game_id) -> Hand | None:
    return db.scalars(
        select(Hand).where(Hand.game_id == game_id).order_by(desc(Hand.hand_index)).limit(1)
    ).first()


@app.post("/games/{game_id}/act", response_model=ActResponse)
def act(game_id: str, payload: ActionRequest, db: Session = Depends(get_db)):
    if policy_engine is None:
        raise HTTPException(status_code=500, detail="Policy engine not loaded")

    game = db.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if game.terminal:
        raise HTTPException(status_code=400, detail="Game is already finished")

    hand = get_latest_hand(db, game.id)
    if not hand:
        raise HTTPException(status_code=500, detail="Hand not found")

    state = deserialize_state(game.current_state)
    if state.current_player != game.human_player:
        raise HTTPException(status_code=400, detail="It is not the human player's turn")

    legal_actions = state.legal_actions()
    if payload.action not in legal_actions:
        raise HTTPException(status_code=400, detail=f"Illegal action: {payload.action}")

    action_count = db.scalar(select(func.count()).select_from(Action).where(Action.game_id == game.id))
    action_index = int(action_count or 0)

    before = serialize_state(state)
    next_state = state.next_state(payload.action)
    after = serialize_state(next_state)

    log_action(
        db=db,
        game=game,
        hand=hand,
        action_index=action_index,
        actor=game.human_player,
        action=payload.action,
        legal_actions=legal_actions,
        state_before=before,
        state_after=after,
        confidence=None,
        policy_probs=None,
    )

    human_action_result = ActionResult(
        actor=game.human_player,
        action=payload.action,
        confidence=0.0,
        policy_probs={},
    )

    bot_action_result = None

    # Bot acts if game not terminal and bot is next
    if not next_state.is_terminal() and next_state.current_player == game.bot_player:
        bot_actions = next_state.legal_actions()
        chosen, confidence, probs = policy_engine.sample_action(next_state, game.bot_player, bot_actions)

        bot_before = serialize_state(next_state)
        bot_state = next_state.next_state(chosen)
        bot_after = serialize_state(bot_state)

        log_action(
            db=db,
            game=game,
            hand=hand,
            action_index=action_index + 1,
            actor=game.bot_player,
            action=chosen,
            legal_actions=bot_actions,
            state_before=bot_before,
            state_after=bot_after,
            confidence=confidence,
            policy_probs=probs,
        )
        next_state = bot_state

        bot_action_result = ActionResult(
            actor=game.bot_player,
            action=chosen,
            confidence=confidence,
            policy_probs=probs,
        )

    game.current_state = serialize_state(next_state)
    game.current_player = next_state.current_player
    game.terminal = next_state.terminal
    game.updated_at = datetime.utcnow()

    winner = None
    utility_p0 = None
    utility_p1 = None

    if next_state.is_terminal():
        game.status = "finished"
        winner = next_state.winner if next_state.reason_is_fold else next_state.showdown_winner()
        utility_p0 = next_state.utility(0)
        utility_p1 = next_state.utility(1)

        hand.finished_at = datetime.utcnow()
        hand.winner = winner
        hand.final_pot = next_state.pot
        hand.result = "fold" if next_state.reason_is_fold else "showdown"
        hand.final_state = serialize_state(next_state)

    db.add(game)
    db.add(hand)
    db.commit()

    public_state = public_state_view(next_state, game.human_player)

    return ActResponse(
        game_id=str(game.id),
        hand_id=str(hand.id),
        state=public_state,
        human_action=human_action_result,
        bot_action=bot_action_result,
        terminal=next_state.is_terminal(),
        winner=winner,
        utility_p0=utility_p0,
        utility_p1=utility_p1,
    )


@app.post("/games/{game_id}/new-hand", response_model=NewHandResponse)
def new_hand(game_id: str, db: Session = Depends(get_db)):
    game = db.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if not game.terminal:
        raise HTTPException(status_code=400, detail="Current hand is not finished")

    state = new_game_state()
    latest = get_latest_hand(db, game.id)
    next_index = 0 if latest is None else latest.hand_index + 1
    hand = Hand(game=game, hand_index=next_index)

    game.current_state = serialize_state(state)
    game.current_player = state.current_player
    game.terminal = state.terminal
    game.status = "active"
    game.updated_at = datetime.utcnow()

    db.add(hand)
    db.add(game)
    db.commit()
    db.refresh(hand)

    public_state = public_state_view(state, game.human_player)
    return NewHandResponse(game_id=str(game.id), hand_id=str(hand.id), state=public_state)


@app.get("/games/{game_id}", response_model=ActResponse)
def get_game(game_id: str, db: Session = Depends(get_db)):
    game = db.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    hand = get_latest_hand(db, game.id)
    if not hand:
        raise HTTPException(status_code=500, detail="Hand not found")

    state = deserialize_state(game.current_state)
    public_state = public_state_view(state, game.human_player)

    winner = None
    utility_p0 = None
    utility_p1 = None
    if state.is_terminal():
        winner = state.winner if state.reason_is_fold else state.showdown_winner()
        utility_p0 = state.utility(0)
        utility_p1 = state.utility(1)

    return ActResponse(
        game_id=str(game.id),
        hand_id=str(hand.id),
        state=public_state,
        terminal=state.is_terminal(),
        winner=winner,
        utility_p0=utility_p0,
        utility_p1=utility_p1,
    )


@app.get("/games/{game_id}/actions", response_model=ActionsResponse)
def get_actions(game_id: str, hand_only: bool = True, db: Session = Depends(get_db)):
    game = db.get(Game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    hand = get_latest_hand(db, game.id)
    if not hand:
        raise HTTPException(status_code=500, detail="Hand not found")

    stmt = select(Action).where(Action.game_id == game.id)
    if hand_only:
        stmt = stmt.where(Action.hand_id == hand.id)
    stmt = stmt.order_by(Action.action_index)
    rows = db.scalars(stmt).all()

    actions = [
        ActionLog(
            action_index=row.action_index,
            actor=row.actor,
            action=row.action,
            confidence=row.confidence,
            policy_probs=row.policy_probs,
            round_index=row.round_index,
            pot=row.pot,
            current_bet=row.current_bet,
            num_raises=row.num_raises,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]

    return ActionsResponse(game_id=str(game.id), hand_id=str(hand.id), actions=actions)


@app.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)):
    total_games = db.scalar(select(func.count()).select_from(Game)) or 0
    total_hands = db.scalar(select(func.count()).select_from(Hand)) or 0
    total_actions = db.scalar(select(func.count()).select_from(Action)) or 0

    bot_rows = db.execute(
        select(Action.action, func.count())
        .join(Game, Action.game_id == Game.id)
        .where(Action.actor == Game.bot_player)
        .group_by(Action.action)
    ).all()
    human_rows = db.execute(
        select(Action.action, func.count())
        .join(Game, Action.game_id == Game.id)
        .where(Action.actor == Game.human_player)
        .group_by(Action.action)
    ).all()

    avg_bot_conf = db.scalar(
        select(func.avg(Action.confidence))
        .join(Game, Action.game_id == Game.id)
        .where(Action.actor == Game.bot_player)
    )

    return StatsResponse(
        total_games=int(total_games),
        total_hands=int(total_hands),
        total_actions=int(total_actions),
        bot_action_freq={row[0]: int(row[1]) for row in bot_rows},
        human_action_freq={row[0]: int(row[1]) for row in human_rows},
        avg_bot_confidence=float(avg_bot_conf) if avg_bot_conf is not None else None,
    )
