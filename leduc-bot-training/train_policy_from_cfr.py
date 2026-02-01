# train_policy_from_cfr.py

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from env_leduc import LeducEnv, LeducState
from cfr_tabular import get_average_strategy_for_key, info_key, train_cfr
from deep_repr import encode_infoset, legal_action_mask, action_to_index
from deep_cfr import PolicyNet, INPUT_DIM, OUTPUT_DIM


# ----------------------
# Dataset collection
# ----------------------

class CFRPolicyDataset(Dataset):
    """
    Holds (features, target_policy, mask) samples.
    features: float[INPUT_DIM]
    target_policy: float[3] (over abstract actions)
    mask: float[3] (0 for illegal, 1 for legal)
    """

    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, target_pi, mask = self.samples[idx]
        return x, target_pi, mask


def cfr_avg_strategy_for_state(state: LeducState, player: int, actions: List[str]) -> List[float]:
    """
    Given a state and legal env actions, look up the tabular CFR average strategy
    for that infoset, and convert it to a 3-dim abstract action distribution.
    """
    key = info_key(state, player, actions)
    local = get_average_strategy_for_key(key, actions)
    s = sum(local)
    if s <= 0:
        n = len(actions)
        local = [1.0 / n] * n
    else:
        local = [p / s for p in local]

    # Map local probs over env actions to abstract 3-action probs
    abstract = [0.0, 0.0, 0.0]
    for a, p in zip(actions, local):
        idx = action_to_index(a)  # 0,1,2
        abstract[idx] += p

    # Normalize over legal abstract actions
    mask = legal_action_mask(actions)
    total = sum(abstract[i] for i in range(3) if mask[i] > 0)
    if total > 0:
        for i in range(3):
            if mask[i] > 0:
                abstract[i] /= total
            else:
                abstract[i] = 0.0
    else:
        # Fallback uniform over legal abstract actions
        legal_indices = [i for i in range(3) if mask[i] > 0]
        for i in legal_indices:
            abstract[i] = 1.0 / len(legal_indices)

    return abstract


def collect_cfr_policy_samples(
    num_hands: int = 20000,
    seed: int = 123
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run games where both players use the tabular CFR policy, and collect
    (features, target_policy, mask) samples at each decision point.
    """
    env = LeducEnv(seed=seed)
    rng = random.Random(seed + 1)
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for _ in range(num_hands):
        state = env.new_game()
        while not state.is_terminal():
            player = state.current_player
            actions = state.legal_actions()

            # Encode state
            feats = encode_infoset(state, player)
            x = torch.tensor(feats, dtype=torch.float32)

            # Target policy from tabular CFR
            target_pi_list = cfr_avg_strategy_for_state(state, player, actions)
            target_pi = torch.tensor(target_pi_list, dtype=torch.float32)

            # Mask over abstract actions
            mask_list = legal_action_mask(actions)
            mask = torch.tensor(mask_list, dtype=torch.float32)

            samples.append((x, target_pi, mask))

            # Now select an action according to the tabular CFR policy to continue the hand
            # (so the state distribution matches how CFR plays)
            # We'll sample from local action probs (over env actions)
            key = info_key(state, player, actions)
            local = get_average_strategy_for_key(key, actions)
            s = sum(local)
            if s <= 0:
                n = len(actions)
                local = [1.0 / n] * n
            else:
                local = [p / s for p in local]

            # Sample env action
            r = rng.random()
            cum = 0.0
            chosen = actions[-1]
            for a, p in zip(actions, local):
                cum += p
                if r <= cum:
                    chosen = a
                    break

            state = state.next_state(chosen)

    return samples


# ----------------------
# Training loop
# ----------------------

def train_policy_net_from_cfr(
    num_hands: int = 20000,
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 123
) -> PolicyNet:
    """
    Collect data from the tabular CFR policy and train a PolicyNet to imitate it.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    print(f"Collecting CFR policy samples from {num_hands} hands...")
    samples = collect_cfr_policy_samples(num_hands=num_hands, seed=seed)
    dataset = CFRPolicyDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = PolicyNet(input_dim=INPUT_DIM, hidden_dim=64, output_dim=OUTPUT_DIM)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        total_batches = 0
        for x, target_pi, mask in loader:
            # x: (B, 21)
            # target_pi: (B, 3)
            # mask: (B, 3)
            logits = net(x)  # (B, 3)

            # Masked softmax and cross-entropy
            # Set logits for illegal actions to large negative
            masked_logits = logits + (mask + 1e-8).log()  # log(0) -> -inf for illegal
            log_probs = F.log_softmax(masked_logits, dim=-1)  # (B, 3)

            # Only consider legal actions in the loss: target_pi already 0 on illegal
            loss = -(target_pi * log_probs).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.4f}")

    return net


# ----------------------
# Simple evaluation vs random using the learned net
# ----------------------

def net_policy_fn(net: PolicyNet, state: LeducState, player: int, actions: List[str]) -> List[float]:
    """
    Wrap the trained PolicyNet into a policy_fn compatible with evaluate_cfr.py.
    """
    import torch.nn.functional as F

    feats = encode_infoset(state, player)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1, 21)
    mask_list = legal_action_mask(actions)
    mask = torch.tensor(mask_list, dtype=torch.float32).unsqueeze(0)  # (1, 3)

    with torch.no_grad():
        logits = net(x)  # (1, 3)
        masked_logits = logits + (mask + 1e-8).log()
        probs = F.softmax(masked_logits, dim=-1)[0].tolist()  # length 3

    # Map abstract action probs back to env actions
    # For simplicity: just gather probability mass for each env action's canonical index.
    env_probs = []
    for a in actions:
        idx = action_to_index(a)
        env_probs.append(probs[idx])
    # Normalize in case multiple env actions share an index (rare here)
    s = sum(env_probs)
    if s <= 0:
        n = len(actions)
        return [1.0 / n] * n
    return [p / s for p in env_probs]


if __name__ == "__main__":
    from evaluate_cfr import evaluate_match, random_policy, simple_heuristic_policy, cfr_policy

    print("Training tabular CFR first...")
    train_cfr(iterations=500000)  # or keep 200000 if you prefer

    print("Training PolicyNet to imitate CFR...")
    policy_net = train_policy_net_from_cfr(
        num_hands=50000,
        batch_size=128,
        epochs=15,
        lr=1e-3,
        seed=123,
    )

    # Wrap trained net as a policy_fn
    def learned_policy(state: LeducState, player: int, actions: List[str]) -> List[float]:
        return net_policy_fn(policy_net, state, player, actions)

    print("Evaluating learned PolicyNet vs random...")
    ev_net_vs_random = evaluate_match(learned_policy, random_policy, num_hands=50000)
    print(f"EV (P0) net vs random: {ev_net_vs_random:.4f} chips/hand")

    print("Evaluating learned PolicyNet vs heuristic...")
    ev_net_vs_heur = evaluate_match(learned_policy, simple_heuristic_policy, num_hands=50000)
    print(f"EV (P0) net vs heuristic: {ev_net_vs_heur:.4f} chips/hand")

    print("Evaluating heuristic (P0) vs learned PolicyNet (P1)...")
    ev_heur_P0_vs_net_P1 = evaluate_match(simple_heuristic_policy, learned_policy, num_hands=50000)
    print(f"EV (P0) heuristic vs net: {ev_heur_P0_vs_net_P1:.4f} chips/hand")

    sym_ev_net_vs_heur = 0.5 * (ev_net_vs_heur - ev_heur_P0_vs_net_P1)
    print(f"Symmetric EV net vs heuristic (seat-averaged): {sym_ev_net_vs_heur:.4f} chips/hand")

    print("Evaluating learned PolicyNet (P0) vs CFR (P1)...")
    ev_net_P0_vs_cfr_P1 = evaluate_match(learned_policy, cfr_policy, num_hands=50000)
    print(f"EV (P0) net vs CFR: {ev_net_P0_vs_cfr_P1:.4f} chips/hand")

    print("Evaluating CFR (P0) vs learned PolicyNet (P1)...")
    ev_cfr_P0_vs_net_P1 = evaluate_match(cfr_policy, learned_policy, num_hands=50000)
    print(f"EV (P0) CFR vs net: {ev_cfr_P0_vs_net_P1:.4f} chips/hand")

    sym_ev_net_vs_cfr = 0.5 * (ev_net_P0_vs_cfr_P1 - ev_cfr_P0_vs_net_P1)
    print(f"Symmetric EV net vs CFR (seat-averaged): {sym_ev_net_vs_cfr:.4f} chips/hand")

    # Save the trained PolicyNet
    import torch
    torch.save(policy_net.state_dict(), "policy_net_leduc.pt")
    print("Saved trained PolicyNet to policy_net_leduc.pt")
