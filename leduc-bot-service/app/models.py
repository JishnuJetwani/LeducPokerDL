import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Game(Base):
    __tablename__ = "games"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(20), default="active")

    human_player: Mapped[int] = mapped_column(Integer)
    bot_player: Mapped[int] = mapped_column(Integer)

    current_state: Mapped[dict] = mapped_column(JSONB)
    current_player: Mapped[int] = mapped_column(Integer)
    terminal: Mapped[bool] = mapped_column(Boolean, default=False)

    hands: Mapped[list["Hand"]] = relationship(back_populates="game")
    actions: Mapped[list["Action"]] = relationship(back_populates="game")


class Hand(Base):
    __tablename__ = "hands"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id"))
    hand_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    winner: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_pot: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    game: Mapped[Game] = relationship(back_populates="hands")
    actions: Mapped[list["Action"]] = relationship(back_populates="hand")


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id"))
    hand_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("hands.id"))
    action_index: Mapped[int] = mapped_column(Integer)

    actor: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(10))
    legal_actions: Mapped[dict] = mapped_column(JSONB)
    policy_probs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    round_index: Mapped[int] = mapped_column(Integer)
    pot: Mapped[int] = mapped_column(Integer)
    current_bet: Mapped[int] = mapped_column(Integer)
    num_raises: Mapped[int] = mapped_column(Integer)

    state_before: Mapped[dict] = mapped_column(JSONB)
    state_after: Mapped[dict] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    game: Mapped[Game] = relationship(back_populates="actions")
    hand: Mapped[Hand] = relationship(back_populates="actions")
