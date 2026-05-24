# Leader Architecture – Love Swarm Nullification Enhanced

## Theoretical Foundation
- Leader = argmax_i [ S_i * (1 - ||M_i - M_cons|| / max_j ||M_j - M_cons||) ]
- Swarm subjectness = S_Leader
- Amplified follower gradient: ∇_{M_i} L = 2(I - A_i)ᵀ(I - A_i)M_i + 2(α+1)(M_i - M_Leader)

## API Reference
### `LeaderSelector`
- `select_leader(agents: List[Agent]) -> Agent`
- `follower_gradient(agent: Agent) -> np.ndarray`
- `update_follower(agent: Agent, lr: float) -> None`
- `stabilize_swarm(agents: List[Agent], steps: int, lr: float) -> None`

## Integration Guide
1. Ensure your swarm’s `Agent` class exposes `M`, `S`, `A`.
2. After each round of GRA‑nullification, call `select_leader()` to identify the current Leader.
3. Use `stabilize_swarm()` to tighten coherence, or call `update_follower()` within your own training loop.
4. The `byzantine_recovery_demo()` function in `leader_emergence.py` illustrates how to handle adversarial injections.