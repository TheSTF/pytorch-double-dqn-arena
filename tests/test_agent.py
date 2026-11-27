import numpy as np
import pytest
import torch

from agent.DQNAgent import DQNAgent, QNetwork


def test_qnetwork_output_shape_single_and_batch():
    net = QNetwork(state_size=(14,), action_size=4, hidden_dim=128)

    single = np.zeros(14, dtype=np.float32)
    out_single = net(single)
    assert out_single.shape == (1, 4)

    batch = torch.zeros((16, 14), dtype=torch.float32)
    out_batch = net(batch)
    assert out_batch.shape == (16, 4)


def test_choose_action_bounds():
    agent = DQNAgent(state_shape=(14,), action_size=4)
    state = np.zeros(14, dtype=np.float32)

    for eps in (0.0, 1.0):
        action = agent.choose_action(state, epsilon=eps)
        assert isinstance(action, int)
        assert 0 <= action < 4


def test_learn_updates_weights_and_syncs_target_network():
    torch.manual_seed(0)
    agent = DQNAgent(
        state_shape=(4,),
        action_size=3,
        memory_capacity=100,
        target_update_freq=2,
        hidden_dim=32,
    )

    # Pas assez de transitions -> retourne None
    assert agent.learn(batch_size=8) is None

    for i in range(16):
        s = np.full(4, float(i) / 16.0, dtype=np.float32)
        ns = np.full(4, float(i + 1) / 16.0, dtype=np.float32)
        agent.remember(s, i % 3, 1.0, ns, i == 15)

    # 1re mise à jour : policy network change, target network reste figé
    loss1 = agent.learn(batch_size=8, gamma=0.99)
    assert isinstance(loss1, float) and np.isfinite(loss1)
    assert any(
        not torch.equal(p_q, p_t)
        for p_q, p_t in zip(agent.q_network.parameters(), agent.target_network.parameters())
    )

    # 2e mise à jour (target_update_freq=2) : target network resynchronisé
    loss2 = agent.learn(batch_size=8, gamma=0.99)
    assert isinstance(loss2, float) and np.isfinite(loss2)
    assert all(
        torch.equal(p_q, p_t)
        for p_q, p_t in zip(agent.q_network.parameters(), agent.target_network.parameters())
    )


def test_save_and_load_roundtrip(tmp_path):
    agent = DQNAgent(state_shape=(14,), action_size=4, hidden_dim=64)
    checkpoint = tmp_path / "test_model.pth"
    agent.save(checkpoint)

    loaded = DQNAgent(state_shape=(14,), action_size=4, hidden_dim=64)
    loaded.load(checkpoint)

    for p1, p2 in zip(agent.q_network.parameters(), loaded.q_network.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(loaded.q_network.parameters(), loaded.target_network.parameters()):
        assert torch.equal(p1, p2)