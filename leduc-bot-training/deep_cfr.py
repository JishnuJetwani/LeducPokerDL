# deep_cfr.py

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_leduc import LeducEnv, LeducState
from deep_repr import encode_infoset, legal_action_mask

INPUT_DIM = 21   # from encode_infoset design
OUTPUT_DIM = 3   # 0: check/call, 1: bet/raise, 2: fold


class RegretNet(nn.Module):
    # Predicts per-action advantages/regrets given an infoset encoding.
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 64, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.out(h)  # (batch, output_dim)


class PolicyNet(nn.Module):
    # Predicts a policy (action probabilities) given an infoset encoding.
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 64, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.out(h)  # (batch, output_dim)
        return logits


def demo_one_state():
    """
    Sanity check:
      - create a fresh Leduc state,
      - encode it,
      - run through RegretNet & PolicyNet,
      - print shapes and a masked softmax policy.
    """
    env = LeducEnv(seed=0)
    state: LeducState = env.new_game()
    player = 0
    actions = state.legal_actions()

    # Encode state
    feats_list: List[float] = encode_infoset(state, player)
    x = torch.tensor(feats_list, dtype=torch.float32).unsqueeze(0)  # (1, input_dim)

    # Legal action mask (over abstract 3 actions)
    mask_list = legal_action_mask(actions)
    mask = torch.tensor(mask_list, dtype=torch.float32).unsqueeze(0)  # (1, 3)

    # Init nets
    regret_net = RegretNet()
    policy_net = PolicyNet()

    # Forward pass
    with torch.no_grad():
        advantages = regret_net(x)  # (1, 3)
        logits = policy_net(x)      # (1, 3)

        # Masked softmax for policy
        # Set logits for illegal actions to large negative
        masked_logits = logits + (mask + 1e-8).log()  # log(0) -> -inf for illegal
        probs = F.softmax(masked_logits, dim=-1)

    print("State:", state)
    print("Legal env actions:", actions)
    print("Encoded features shape:", x.shape)
    print("Advantages:", advantages.numpy())
    print("Raw logits:", logits.numpy())
    print("Mask:", mask.numpy())
    print("Masked policy probs over [check/call, bet/raise, fold]:", probs.numpy())


if __name__ == "__main__":
    demo_one_state()
