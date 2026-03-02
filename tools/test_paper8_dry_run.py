#!/usr/bin/env python3
"""
Tiny dry-run validator for Paper 8 components (no full experiments).

Validates:
- Topology generator produces required attrs
- Paths + allocation contexts generate cleanly
- Environment can compute physics-based rewards without empty reward_functions
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]  # quantum_project_hub/
    sys.path.insert(0, str(repo_root))

    try:
        import numpy as np
    except ModuleNotFoundError:
        print("⏭️  SKIP: numpy not installed in this python environment")
        return 0

    import networkx as nx

    from daqr.core.attack_strategy import NoAttack
    from daqr.core.network_environment import QuantumEnvironment
    from daqr.core.quantum_physics import Paper8NoiseModel, Paper8FidelityCalculator
    from daqr.core.topology_generator import Paper8RandomConnectedTopologyGenerator

    rng = np.random.default_rng(42)

    topo = Paper8RandomConnectedTopologyGenerator(
        num_nodes=25,
        connection_prob=0.12,
        seed=42,
        fidelity_range=(0.7, 0.95),
        rate_range=(0.7, 1.0),
        pur_round_range=(0, 4),
        swap_success_range=(0.8, 0.99),
    ).generate()

    # Quick attribute sanity
    u, v = next(iter(topo.edges()))
    assert "fidelity" in topo[u][v]
    assert "rate" in topo[u][v]
    assert "pur_round" in topo[u][v]
    assert "swap_success" in topo.nodes[next(iter(topo.nodes()))]

    # Create a small set of candidate paths
    nodes = list(topo.nodes())
    paths = []
    while len(paths) < 4:
        src, dst = rng.choice(nodes, 2, replace=False)
        try:
            p = nx.shortest_path(topo, src, dst, weight="distance")
        except nx.NetworkXNoPath:
            continue
        if p not in paths and len(p) >= 2:
            paths.append(p)

    # Per-path budgets (same count as paths)
    qubit_cap = (5, 5, 4, 4)

    # Allocation contexts per path: compositions of cap into hop_count parts
    def compositions(total: int, parts: int, limit: int = 2000):
        out = []

        def rec(remaining: int, k: int, prefix):
            if len(out) >= limit:
                return
            if k == 1:
                out.append(prefix + [remaining])
                return
            for x in range(remaining + 1):
                if len(out) >= limit:
                    return
                rec(remaining - x, k - 1, prefix + [x])

        rec(int(total), int(parts), [])
        return out

    external_contexts = []
    for cap, path in zip(qubit_cap, paths):
        hops = max(1, len(path) - 1)
        external_contexts.append(np.array(compositions(cap, hops), dtype=int))

    noise_model = Paper8NoiseModel(topology=topo, paths=paths)
    fidelity_calc = Paper8FidelityCalculator(min_fidelity=0.0)

    env = QuantumEnvironment(
        attack=NoAttack(),
        qubit_capacities=qubit_cap,
        allocator=None,
        noise_model=noise_model,
        fidelity_calculator=fidelity_calc,
        external_contexts=external_contexts,
        external_rewards=None,
        external_topology=topo,
        frame_length=10,
        horizon_length=10,
        num_paths=len(qubit_cap),
        num_total_qubits=sum(qubit_cap),
        seed=42,
    )

    info = env.get_environment_info()
    rewards = info.get("reward_functions")
    assert rewards and len(rewards) == len(qubit_cap), "reward_functions missing or wrong length"
    assert all(len(r) > 0 for r in rewards), "some path reward lists are empty"

    print("✅ Paper8 dry-run OK")
    print(f"   Topology: {topo.number_of_nodes()} nodes, {topo.number_of_edges()} edges")
    print(f"   Paths: {len(paths)}")
    print(f"   Rewards lens: {[len(r) for r in rewards]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
