# Leduc CFR Service

FastAPI + Postgres service that serves a pretrained Leduc Hold'em CFR policy and logs game analytics.

## Local Dev (Docker)

From the `leduc-bot-service` directory:

```bash
docker compose up --build
```

Then in another terminal (inside the same directory):

```bash
docker compose exec api alembic revision --autogenerate -m "init"
docker compose exec api alembic upgrade head
```

API runs at `http://localhost:8000`.

## API Basics

- `POST /games` with `{ "human_player": 0 }` to start a game.
- `POST /games/{game_id}/act` with `{ "action": "check" }` to play your move.
- `POST /games/{game_id}/new-hand` to start a new hand after the current one ends.
- `GET /games/{game_id}` to fetch current state.
- `GET /games/{game_id}/actions` to fetch logged actions for the current hand.
- `GET /stats` to fetch basic analytics.
- `GET /health` for health check.

## Notes

- The service imports training code from `../leduc-bot-training` and loads `policy_net_leduc.pt`.
- For production, swap `DATABASE_URL` to point at RDS and run Alembic migrations.

## Deployment Notes (Lightsail + RDS)

1) Create an RDS Postgres instance and note the endpoint, user, password, DB name.
2) Open RDS security group to allow inbound from your Lightsail instance.
3) Create a Lightsail instance (Ubuntu) and install Docker.
4) Copy this repo to the instance and set `DATABASE_URL` to the RDS endpoint, then run:

```bash
docker compose up -d --build
docker compose exec api alembic upgrade head
```

5) Expose port 8000 in Lightsail networking and point your domain or test via public IP.
