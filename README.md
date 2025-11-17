# Leduc Hold’em Poker Bot

A compact poker AI project built on Leduc Hold’em, a standard toy game in game-theory and RL research. 
- Implements a full 2-player fixed-limit Leduc environment.
- Trains a tabular CFR solver to approximate a Nash equilibrium strategy.
- Trains a neural PolicyNet to imitate the CFR strategy.
- Provides a CLI to play against the trained model.

Features:

- Game engine
  2-player fixed-limit Leduc Hold’em (`env_leduc.py`).

- Solver: Tabular CFR
  Counterfactual Regret Minimization over 288 information sets (`cfr_tabular.py`).

- Baselines
  Random and simple heuristic strategies for comparison (`evaluate_cfr.py`).

- Neural policy
  A 2-layer MLP PolicyNet using a 21-dimensional encoding of hand, board, and betting history (`deep_repr.py`, `deep_cfr.py`).

- Training pipeline 
  CFR → generate self-play samples → supervised learning → evaluation (`train_policy_from_cfr.py`).

- Interactive play
  Human-vs-bot terminal interface using the trained policy (`human_vs_bot.py`).

---

Example Results:

With tabular CFR trained for 500,000 iterations and the PolicyNet trained on 50,000 CFR self-play hands:

- CFR (P0) vs random: ~+0.60 chips/hand
- PolicyNet (P0) vs random: ~+0.55 chips/hand
  *PolicyNet vs heuristic (seat-averaged): ~+0.07 chips/hand
- PolicyNet vs CFR (seat-averaged): ~−0.003 chips/hand

The neural policy:

- Clearly beats random and a simple human-style heuristic.
- Tracks the CFR baseline very closely in symmetric EV.

---

## Project Structure

```text
.
├── env_leduc.py              # LeducState + LeducEnv (rules, transitions)
├── cfr_tabular.py            # Tabular CFR solver (infosets, regret matching, train_cfr)
├── evaluate_cfr.py           # Policy wrappers + EV evaluation vs random/heuristic
├── deep_repr.py              # 21-dim encoding + 3-action mask (check/call, bet/raise, fold)
├── deep_cfr.py               # PolicyNet (MLP) and related utilities
├── train_policy_from_cfr.py  # Pipeline: CFR → PolicyNet_
