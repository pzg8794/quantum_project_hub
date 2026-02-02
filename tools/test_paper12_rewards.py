#!/usr/bin/env python3
"""
Quick test to verify Paper 12 reward generation is working.
"""
import sys
sys.path.insert(0, '/Users/pitergarcia/DataScience/Semester4/GA-Work/hybrid_variable_framework/Dynamic_Routing_Eval_Framework')

# Import the notebook's functions
from daqr.quantum_network_sim.paper12_utils import *
from daqr.core.quantum_physics import *
import numpy as np

# Minimal FRAMEWORK_CONFIG for testing
FRAMEWORK_CONFIG = {
    'paper12': {
        'testbed': 'paper12',
        'topology_type': 'waxman_qcast',
        'n_nodes': 100,
        'avg_degree': 6,
        'entanglement_prob': 0.6,
        'fusion_prob': 0.9,
        'epoch_length': 500,
        'total_timeslots': 5000,
        'split_constant': 3,
        'use_dynamic_thresholding': True,
        'num_paths': 4,
        'total_qubits': 100,
        'exploration_bonus': 1.5,
    }
}

print("=" * 80)
print("PAPER 12 REWARD GENERATION TEST")
print("=" * 80)

# Test 1: Generate physics params
config = FRAMEWORK_CONFIG['paper12']
print("\n1. Generating Paper 12 Physics Parameters...")
try:
    physics_params = get_physics_params_paper12(config, seed=42, qubit_cap=None)
    print("✅ Physics parameters generated successfully")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check reward structure
print("\n2. Analyzing Reward Structure...")
rewards = physics_params.get('external_rewards', None)
if rewards is None:
    print("❌ No external rewards found!")
    sys.exit(1)

print(f"✅ Reward lists: {len(rewards)} paths")
for i, path_rewards in enumerate(rewards):
    total = sum(path_rewards)
    avg = total / len(path_rewards)
    min_r = min(path_rewards)
    max_r = max(path_rewards)
    print(f"   Path {i}: {len(path_rewards):2d} arms | "
          f"Min: {min_r:5.1f} | Max: {max_r:5.1f} | Avg: {avg:5.1f} | Sum: {total:6.1f}")

# Test 3: Verify rewards are non-zero
total_reward = sum(sum(path) for path in rewards)
if total_reward == 0:
    print("\n❌ ERROR: Total reward is zero!")
    sys.exit(1)
else:
    print(f"\n✅ Total aggregate reward: {total_reward:.1f} (non-zero)")

# Test 4: Check contexts align with rewards
print("\n3. Checking Context-Reward Alignment...")
contexts = physics_params.get('external_contexts', None)
if contexts is None:
    print("❌ No external contexts found!")
    sys.exit(1)

print(f"✅ Context arrays: {len(contexts)} paths")
if len(contexts) != len(rewards):
    print(f"❌ Context count ({len(contexts)}) doesn't match reward count ({len(rewards)})")
    sys.exit(1)

for i, (ctx, rew) in enumerate(zip(contexts, rewards)):
    if ctx.shape[0] != len(rew):
        print(f"❌ Path {i}: context arms ({ctx.shape[0]}) != reward arms ({len(rew)})")
        sys.exit(1)
    print(f"   Path {i}: {ctx.shape[0]} arms → {len(rew)} rewards ✓")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - Paper 12 rewards are properly generated!")
print("=" * 80)
