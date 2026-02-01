import React, { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";
const STARTING_STACK = 10;

function formatHistory(history) {
  if (!history || history.length === 0) return "No actions yet.";
  return history
    .map(([round, player, action]) => `R${round} - P${player}: ${action}`)
    .join("\n");
}

export default function App() {
  const [gameId, setGameId] = useState(null);
  const [handId, setHandId] = useState(null);
  const [state, setState] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [lastBot, setLastBot] = useState(null);
  const [lastHuman, setLastHuman] = useState(null);
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(null);
  const [humanSeat, setHumanSeat] = useState(0);

  const legalActions = useMemo(() => state?.legal_actions || [], [state]);
  const roundLabel = state?.round_index === 0 ? "Preflop" : "Flop";
  const pot = state?.pot ?? 0;
  const totalBets = state?.total_bets || [0, 0];
  const humanIndex = humanSeat;
  const botIndex = 1 - humanSeat;
  const humanCommitted = totalBets[humanIndex] ?? 0;
  const botCommitted = totalBets[botIndex] ?? 0;
  const humanStack = Math.max(STARTING_STACK - humanCommitted, 0);
  const botStack = Math.max(STARTING_STACK - botCommitted, 0);

  async function createGame(humanPlayer = 0) {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/games`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ human_player: humanPlayer })
      });
      if (!res.ok) throw new Error("Failed to create game");
      const data = await res.json();
      setGameId(data.game_id);
      setHandId(data.hand_id);
      setState(data.state);
      setHumanSeat(humanPlayer);
      setLastBot(null);
      setLastHuman(null);
      setLogs([]);
      await fetchStats();
    } catch (err) {
      setError(err.message || "Unknown error");
    } finally {
      setBusy(false);
    }
  }

  async function act(action) {
    if (!gameId) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/games/${gameId}/act`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action })
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to act");
      }
      const data = await res.json();
      setState(data.state);
      setLastHuman(data.human_action || null);
      setLastBot(data.bot_action || null);
      await fetchLogs(data.game_id);
      await fetchStats();
    } catch (err) {
      setError(err.message || "Unknown error");
    } finally {
      setBusy(false);
    }
  }

  async function fetchLogs(id = gameId) {
    if (!id) return;
    try {
      const res = await fetch(`${API_BASE}/games/${id}/actions`);
      if (!res.ok) return;
      const data = await res.json();
      setLogs(data.actions || []);
      setHandId(data.hand_id || handId);
    } catch {
      // ignore log errors
    }
  }

  async function fetchStats() {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (!res.ok) return;
      const data = await res.json();
      setStats(data);
    } catch {
      // ignore stats errors
    }
  }

  async function newHand() {
    if (!gameId) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/games/${gameId}/new-hand`, {
        method: "POST"
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to start new hand");
      }
      const data = await res.json();
      setHandId(data.hand_id);
      setState(data.state);
      setLastBot(null);
      setLastHuman(null);
      setLogs([]);
      await fetchStats();
    } catch (err) {
      setError(err.message || "Unknown error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Leduc Hold’em Demo</p>
          <h1>Play a two-round poker game against a CFR agent.</h1>
          <p className="subtitle">
            Leduc Hold’em is a simplified poker variant with a 6-card deck (J, Q, K in two suits),
            two betting rounds, and a single public card on the flop. Pairs beat high cards,
            and the CFR bot plays an approximate equilibrium strategy.
          </p>
          <div className="actions">
            <button disabled={busy} onClick={() => createGame(0)}>
              Start Hand (You = P0)
            </button>
            <button disabled={busy} onClick={() => createGame(1)}>
              Start Hand (You = P1)
            </button>
            <button disabled={busy || !state?.terminal} onClick={newHand}>
              New Hand
            </button>
          </div>
        </div>
        <div className="hero-card">
          <div className="metric">
            <span>Pot</span>
            <strong>{pot}</strong>
          </div>
          <div className="metric">
            <span>Round</span>
            <strong>{state ? roundLabel : "—"}</strong>
          </div>
          <div className="metric">
            <span>Public</span>
            <strong>{state?.public_card || "—"}</strong>
          </div>
          <div className="metric">
            <span>To Call</span>
            <strong>{state ? state.current_bet : "—"}</strong>
          </div>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <main className="table-layout">
        <section className="player bot">
          <div className="seat-header">
            <h2>Bot</h2>
            <span>Seat P{botIndex}</span>
          </div>
          <div className="chips">
            <div>Stack: {botStack}</div>
            <div>Committed: {botCommitted}</div>
          </div>
          <div className="cards">
            <div className="card back">?</div>
          </div>
          {lastBot && (
            <div className="last-action">
              <span>Last:</span> {lastBot.action} ({lastBot.confidence.toFixed(2)})
            </div>
          )}
        </section>

        <section className="table-center">
          <div className="board">
            <div className="board-label">Board</div>
            <div className="cards">
              <div className={`card ${state?.public_card ? "face" : "back"}`}>
                {state?.public_card || "—"}
              </div>
            </div>
          </div>
          <div className="pot-chip">Pot {pot}</div>
          <div className="history-panel">
            <h3>Action History</h3>
            <pre className="history">{formatHistory(state?.history || [])}</pre>
          </div>
        </section>

        <section className="player human">
          <div className="seat-header">
            <h2>You</h2>
            <span>Seat P{humanIndex}</span>
          </div>
          <div className="chips">
            <div>Stack: {humanStack}</div>
            <div>Committed: {humanCommitted}</div>
          </div>
          <div className="cards">
            <div className="card face">{state?.private_card || "—"}</div>
          </div>
          {lastHuman && (
            <div className="last-action">
              <span>Last:</span> {lastHuman.action}
            </div>
          )}
          <div className="action-list">
            {(!state || legalActions.length === 0) && <p className="muted">Start a hand to act.</p>}
            {legalActions.map((action) => (
              <button
                key={action}
                disabled={busy || state?.terminal || state?.current_player !== humanIndex}
                onClick={() => act(action)}
              >
                {action}
              </button>
            ))}
          </div>
        </section>
      </main>

      <section className="analytics">
        <div className="panel">
          <h2>Action Log</h2>
          {logs.length === 0 ? (
            <p>No actions logged yet.</p>
          ) : (
            <div className="log-list">
              {logs.map((row) => (
                <div key={`${row.action_index}-${row.action}`} className="log-row">
                  <div>
                    <strong>P{row.actor}</strong> {row.action}
                  </div>
                  <div className="muted">
                    R{row.round_index} • pot {row.pot} • bet {row.current_bet}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="panel">
          <h2>Stats</h2>
          {stats ? (
            <div className="state">
              <div><span>Total games:</span> {stats.total_games}</div>
              <div><span>Total hands:</span> {stats.total_hands}</div>
              <div><span>Total actions:</span> {stats.total_actions}</div>
              <div><span>Avg bot confidence:</span> {stats.avg_bot_confidence?.toFixed(3) ?? "n/a"}</div>
            </div>
          ) : (
            <p>Stats unavailable.</p>
          )}
        </div>
      </section>
    </div>
  );
}
