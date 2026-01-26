import random
import socket
import sys
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Device setup (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN hyperparameters
MEMORY_SIZE = 50000
MEMORY_WARMUP_SIZE = 500
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPS_INIT = 0.1
EPS_MIN = 0.01
EPS_DEC = 1e-6
TARGET_SYNC_STEPS = 500

# Observation layout
STACK = 3
FEAT = 5

# Analytic action catalog (kept identical to cipd_bi-dqn.py)
THETAS_M = [0.16, 0.18, 0.20, 0.22]
GAPS = [0.08, 0.10, 0.12]
ALPHAS = [0.05, 0.10, 0.20]
OFFSETS = {
    "A": (1, 1, 1),
    "B": (1, 2, 4),
}
ACTION_CATALOG = []
for tm in THETAS_M:
    for g in GAPS:
        th = min(tm + g, 0.35)
        for aew in ALPHAS:
            for key in ["A", "B"]:
                ACTION_CATALOG.append((tm, th, aew, *OFFSETS[key]))
ACT_DIM = len(ACTION_CATALOG)


class ReplayMemory:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        obs, act, rew, next_obs, done = zip(*batch)
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        act = torch.tensor(np.array(act), dtype=torch.int64)
        rew = torch.tensor(np.array(rew), dtype=torch.float32)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32)
        done = torch.tensor(np.array(done), dtype=torch.float32)
        return obs, act, rew, next_obs, done

    def __len__(self):
        return len(self.buffer)


class DuelingBiLSTM(nn.Module):
    """Dueling head stacked on top of a small bi-LSTM encoder."""

    def __init__(self, obs_dim: int, act_dim: int, feat: int = FEAT, stack: int = STACK):
        super().__init__()
        self.obs_dim = obs_dim
        self.feat = feat
        self.stack = stack
        self.trunk_out = 128
        self.hidden = 64

        self.fc_in = nn.Linear(self.feat, self.trunk_out)
        self.bilstm = nn.LSTM(
            input_size=self.trunk_out,
            hidden_size=self.hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out = self.hidden * 2

        self.v_fc = nn.Linear(lstm_out, 128)
        self.v_out = nn.Linear(128, 1)
        self.a_fc = nn.Linear(lstm_out, 128)
        self.a_out = nn.Linear(128, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, obs_dim] flattened slices
        bsz = x.size(0)
        seq = x.view(bsz, self.stack, self.feat)
        seq = seq.reshape(bsz * self.stack, self.feat)
        seq = F.relu(self.fc_in(seq))
        seq = seq.view(bsz, self.stack, self.trunk_out)

        out, _ = self.bilstm(seq)
        last = out[:, -1, :]

        v = F.relu(self.v_fc(last))
        v = self.v_out(v)
        a = F.relu(self.a_fc(last))
        a = self.a_out(a)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class AgentTorch:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = LEARNING_RATE,
        gamma: float = GAMMA,
        eps_init: float = EPS_INIT,
        eps_min: float = EPS_MIN,
        eps_dec: float = EPS_DEC,
        target_sync: int = TARGET_SYNC_STEPS,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.target_sync = target_sync
        self.global_step = 0

        self.device = DEVICE
        self.q = DuelingBiLSTM(obs_dim, act_dim).to(self.device)
        self.target = DuelingBiLSTM(obs_dim, act_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.mem = ReplayMemory(MEMORY_SIZE)

    def sample_action(self, obs_np: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            a = np.random.randint(self.act_dim)
        else:
            with torch.no_grad():
                x = torch.tensor(obs_np[None, :], dtype=torch.float32, device=self.device)
                qvals = self.q(x)
                a = int(torch.argmax(qvals, dim=1).item())
        self.eps = max(self.eps_min, self.eps - self.eps_dec)
        return a

    def predict_action(self, obs_np: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.tensor(obs_np[None, :], dtype=torch.float32, device=self.device)
            qvals = self.q(x)
            return int(torch.argmax(qvals, dim=1).item())

    def learn(self, batch_size: int = BATCH_SIZE):
        if len(self.mem) < batch_size:
            return None
        obs, act, rew, next_obs, done = self.mem.sample(batch_size)
        obs = obs.to(self.device)
        act = act.to(self.device).view(-1, 1)
        rew = rew.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        qvals = self.q(obs).gather(1, act).squeeze(1)
        with torch.no_grad():
            next_qvals = self.target(next_obs).max(1)[0]
            target = rew + (1.0 - done) * self.gamma * next_qvals

        loss = F.smooth_l1_loss(qvals, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.global_step += 1
        if self.global_step % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()


class AgentManager:
    """Lightweight API for C++ to call into."""

    def __init__(self, obs_dim: int = STACK * FEAT, act_dim: int = ACT_DIM, seed: int = 1):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.agents: Dict[str, AgentTorch] = {
            "A": AgentTorch(obs_dim, act_dim),
            "B": AgentTorch(obs_dim, act_dim),
        }

    def _agent(self, agent_id: str) -> AgentTorch:
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent id {agent_id}")
        return self.agents[agent_id]

    def act(self, agent_id: str, obs: List[float]) -> int:
        if len(obs) != self.obs_dim:
            raise ValueError(f"obs length {len(obs)} != expected {self.obs_dim}")
        return self._agent(agent_id).sample_action(np.array(obs, dtype=np.float32))

    def predict(self, agent_id: str, obs: List[float]) -> int:
        if len(obs) != self.obs_dim:
            raise ValueError(f"obs length {len(obs)} != expected {self.obs_dim}")
        return self._agent(agent_id).predict_action(np.array(obs, dtype=np.float32))

    def store(self, agent_id: str, obs: List[float], action: int, reward: float, next_obs: List[float], done: float):
        if len(obs) != self.obs_dim or len(next_obs) != self.obs_dim:
            raise ValueError("Observation size mismatch")
        self._agent(agent_id).mem.append(
            (
                np.array(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_obs, dtype=np.float32),
                float(done),
            )
        )
        return len(self._agent(agent_id).mem)

    def learn(self, agent_id: str, batch_size: int = BATCH_SIZE):
        return self._agent(agent_id).learn(batch_size=batch_size)

    def mem_len(self, agent_id: str) -> int:
        return len(self._agent(agent_id).mem)


def parse_floats(tokens):
    return [float(t) for t in tokens]


def run_socket_client(host: str = "127.0.0.1", port: int = 5555):
    """Connect to main.cc (server) and serve RL requests over a simple line protocol."""
    mgr = AgentManager()
    obs_dim = mgr.obs_dim
    act_dim = mgr.act_dim

    with socket.create_connection((host, port)) as sock:
        f = sock.makefile("r")
        out = sock.makefile("w")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].upper()

            try:
                if cmd in ("INIT", "HELLO"):
                    # Expect INIT <obs_dim> <act_dim>
                    if len(parts) >= 3:
                        exp_obs = int(parts[1])
                        exp_act = int(parts[2])
                        if exp_obs != obs_dim or exp_act != act_dim:
                            raise ValueError(f"Dim mismatch: expected ({obs_dim},{act_dim}) got ({exp_obs},{exp_act})")
                    out.write("OK\n")
                    out.flush()
                elif cmd == "RESET":
                    mgr = AgentManager()
                    out.write("OK\n")
                    out.flush()
                elif cmd in ("ACT", "PREDICT"):
                    agent_id = parts[1]
                    obs = parse_floats(parts[2:])
                    if len(obs) != obs_dim:
                        raise ValueError(f"obs length {len(obs)} != {obs_dim}")
                    action = mgr.act(agent_id, obs) if cmd == "ACT" else mgr.predict(agent_id, obs)
                    out.write(f"ACTION {action}\n")
                    out.flush()
                elif cmd == "STORE":
                    agent_id = parts[1]
                    action = int(parts[2])
                    reward = float(parts[3])
                    done = float(parts[4])
                    obs_tokens = parts[5 : 5 + obs_dim]
                    next_tokens = parts[5 + obs_dim : 5 + obs_dim + obs_dim]
                    obs = parse_floats(obs_tokens)
                    next_obs = parse_floats(next_tokens)
                    mem_len = mgr.store(agent_id, obs, action, reward, next_obs, done)
                    out.write(f"MEM {mem_len}\n")
                    out.flush()
                elif cmd == "LEARN":
                    agent_id = parts[1]
                    batch_size = int(parts[2])
                    loss = mgr.learn(agent_id, batch_size=batch_size)
                    if loss is None:
                        out.write("LOSS nan\n")
                    else:
                        out.write(f"LOSS {loss}\n")
                    out.flush()
                elif cmd == "MEMLEN":
                    agent_id = parts[1]
                    out.write(f"MEM {mgr.mem_len(agent_id)}\n")
                    out.flush()
                elif cmd == "CLOSE":
                    out.write("BYE\n")
                    out.flush()
                    break
                else:
                    out.write("ERR unknown_cmd\n")
                    out.flush()
            except Exception as exc:  # keep session alive on bad input
                out.write(f"ERR {type(exc).__name__}:{exc}\n")
                out.flush()


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5555
    if len(sys.argv) >= 2:
        host = sys.argv[1]
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])
    print(f"Connecting to {host}:{port} ...", file=sys.stderr)
    run_socket_client(host, port)
