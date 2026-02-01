import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from app.config import MODEL_PATH, TRAINING_DIR

# Ensure training code is importable
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

from deep_cfr import PolicyNet, INPUT_DIM, OUTPUT_DIM  # type: ignore
from deep_repr import encode_infoset, legal_action_mask, action_to_index  # type: ignore


class PolicyEngine:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.net = PolicyNet(input_dim=INPUT_DIM, hidden_dim=64, output_dim=OUTPUT_DIM)
        state_dict = torch.load(model_path, map_location="cpu")
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def action_probs(self, state, player: int, actions: List[str]) -> Dict[str, float]:
        feats = encode_infoset(state, player)
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        mask_list = legal_action_mask(actions)
        mask = torch.tensor(mask_list, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.net(x)
            masked_logits = logits + (mask + 1e-8).log()
            probs = F.softmax(masked_logits, dim=-1)[0].tolist()

        env_probs = {}
        for a in actions:
            idx = action_to_index(a)
            env_probs[a] = probs[idx]

        total = sum(env_probs.values())
        if total <= 0:
            uniform = 1.0 / len(actions)
            return {a: uniform for a in actions}
        return {a: p / total for a, p in env_probs.items()}

    def sample_action(self, state, player: int, actions: List[str]) -> Tuple[str, float, Dict[str, float]]:
        probs = self.action_probs(state, player, actions)
        r = torch.rand(1).item()
        cum = 0.0
        chosen = actions[-1]
        for a in actions:
            cum += probs[a]
            if r <= cum:
                chosen = a
                break
        confidence = probs[chosen]
        return chosen, confidence, probs
