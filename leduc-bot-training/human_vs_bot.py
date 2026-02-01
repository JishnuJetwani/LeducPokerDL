# human_vs_bot.py

from typing import List
import torch
import torch.nn.functional as F

from env_leduc import LeducEnv, LeducState
from deep_repr import encode_infoset, legal_action_mask
from deep_repr import action_to_index  # if you want it, not strictly needed
from deep_cfr import PolicyNet, INPUT_DIM, OUTPUT_DIM


def load_policy_net(path: str = "policy_net_leduc.pt") -> PolicyNet:
    net = PolicyNet(input_dim=INPUT_DIM, hidden_dim=64, output_dim=OUTPUT_DIM)
    state_dict = torch.load(path, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    return net


def bot_action(net: PolicyNet, state: LeducState, player: int, actions: List[str]) -> str:
    # Choose an action for the bot by sampling from the PolicyNet's distribution.
    feats = encode_infoset(state, player)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1, 21)
    mask_list = legal_action_mask(actions)
    mask = torch.tensor(mask_list, dtype=torch.float32).unsqueeze(0)  # (1, 3)

    with torch.no_grad():
        logits = net(x)  # (1, 3)
        masked_logits = logits + (mask + 1e-8).log()
        probs = F.softmax(masked_logits, dim=-1)[0].tolist()  # length 3

    # Map abstract probs to env actions
    # For this project, each env action has a unique abstract index, so we just copy.
    env_probs = []
    for a in actions:
        idx = action_to_index(a)
        env_probs.append(probs[idx])

    # Normalize in case of any numerical issues
    s = sum(env_probs)
    if s <= 0:
        n = len(actions)
        env_probs = [1.0 / n] * n
    else:
        env_probs = [p / s for p in env_probs]

    # Sample an action according to env_probs
    import random
    r = random.random()
    cum = 0.0
    chosen = actions[-1]
    for a, p in zip(actions, env_probs):
        cum += p
        if r <= cum:
            chosen = a
            break
    return chosen


def print_state(state: LeducState, human_player: int):
    print("\n=== Current State ===")
    print(f"Round index: {state.round_index} (0=preflop, 1=flop)")
    print(f"Pot: {state.pot}, current bet this round: {state.current_bet}")
    print(f"Your private card (Player {human_player}): {state.private_cards[human_player]}")
    if state.public_card is None:
        print("Board: [no public card yet]")
    else:
        print(f"Board card: {state.public_card}")

    print("History (round, player, action):")
    if not state.history:
        print("  [empty]")
    else:
        for (r, p, a) in state.history:
            print(f"  R{r} - P{p}: {a}")


def get_human_action(state: LeducState, human_player: int) -> str:
    actions = state.legal_actions()
    print_state(state, human_player)
    print(f"Legal actions: {', '.join(actions)}")

    while True:
        a = input("Your action: ").strip().lower()
        # Allow some aliases (e.g., 'c' for check/call, 'b' for bet, 'r' for raise, 'f' for fold)
        alias_map = {
            "c": "check" if "check" in actions else "call",
            "b": "bet",
            "r": "raise",
            "f": "fold",
        }
        if a in alias_map:
            a = alias_map[a]

        if a in actions:
            return a
        print(f"Invalid action. Please choose one of: {', '.join(actions)}")


def play_one_hand(net: PolicyNet):
    env = LeducEnv()  # default seed -> random deck each hand
    state = env.new_game()

    # Choose seat
    while True:
        choice = input("Do you want to be Player 0 (acts first preflop) or Player 1? [0/1]: ").strip()
        if choice in ("0", "1"):
            human_player = int(choice)
            break
        print("Please enter 0 or 1.")

    print(f"You are Player {human_player}.")

    # Main hand loop
    while not state.is_terminal():
        current = state.current_player
        actions = state.legal_actions()

        if current == human_player:
            action = get_human_action(state, human_player)
            print(f"You chose: {action}")
        else:
            action = bot_action(net, state, current, actions)
            print(f"Bot (Player {current}) chooses: {action}")

        state = state.next_state(action)

    # Hand is over
    print("\n=== Hand Finished ===")
    print("Final history:")
    for (r, p, a) in state.history:
        print(f"  R{r} - P{p}: {a}")
    print(f"Final pot: {state.pot}")
    util0 = state.utility(0)
    util1 = state.utility(1)
    print(f"Utility P0: {util0}, Utility P1: {util1}")

    if human_player == 0:
        your_util = util0
    else:
        your_util = util1

    if your_util > 0:
        print("You won this hand!")
    elif your_util < 0:
        print("You lost this hand.")
    else:
        print("Hand was a tie.")


def main():
    print("Loading trained PolicyNet from policy_net_leduc.pt ...")
    net = load_policy_net("policy_net_leduc.pt")
    print("Loaded. Let's play Leduc!\n")

    while True:
        play_one_hand(net)
        again = input("\nPlay another hand? [y/n]: ").strip().lower()
        if again not in ("y", "yes"):
            break

    print("Thanks for playing!")


if __name__ == "__main__":
    main()
