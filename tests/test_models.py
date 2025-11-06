import torch

from src.strategies.exper_corr_pos.models import MoEPolicy


def test_gating_weights_sum_to_one_and_mask_respects_topk():
    policy = MoEPolicy(
        input_dim=5,
        num_actions=3,
        expert_hidden=[8],
        gating_hidden=[8],
        num_experts=4,
        top_k=2,
        temperature=0.7,
    )
    obs = torch.randn(32, 5)
    weights, mask = policy.gating(obs, top_k=policy.top_k)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    if policy.top_k < policy.num_experts:
        mask_counts = mask.sum(dim=-1)
        assert torch.all(mask_counts == policy.top_k)


def test_policy_forward_returns_valid_distribution_and_value():
    policy = MoEPolicy(
        input_dim=4,
        num_actions=3,
        expert_hidden=[16, 8],
        gating_hidden=[8, 8],
        num_experts=3,
        top_k=3,
    )
    obs = torch.randn(10, 4)
    dist, value, lb_loss = policy(obs)
    probs = dist.probs
    assert probs.shape == (10, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-6)
    assert value.shape == (10,)
    assert lb_loss >= 0.0


def test_gating_topk_equals_one_and_temperature_low():
    policy = MoEPolicy(
        input_dim=3,
        num_actions=2,
        expert_hidden=[6],
        gating_hidden=[6],
        num_experts=3,
        top_k=1,
        temperature=0.1,
    )
    obs = torch.randn(5, 3)
    weights, mask = policy.gating(obs, top_k=policy.top_k)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    mask_counts = mask.sum(dim=-1)
    assert torch.all(mask_counts == 1)


def test_policy_backward_pass_runs():
    policy = MoEPolicy(
        input_dim=4,
        num_actions=2,
        expert_hidden=[8],
        gating_hidden=[8],
        num_experts=2,
        top_k=2,
    )
    obs = torch.randn(12, 4)
    dist, value, lb_loss = policy(obs)
    probs = dist.probs
    loss = probs.mean() + value.mean() + lb_loss
    loss.backward()
    total_norm = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()
    assert total_norm > 0.0
