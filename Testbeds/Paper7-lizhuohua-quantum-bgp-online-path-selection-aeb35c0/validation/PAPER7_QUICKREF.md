# Paper 7 Integration - Quick Reference Card

## At a Glance

| Item | Status | Location | Action |
|------|--------|----------|--------|
| **Core Classes** | ✅ | `daqr/core/topology_generator.py` | Run diagnostic |
| **Context Generation** | ✅ | `experiments/stochastic_evaluation.py` | Verify ranges |
| **QPSEL Algorithm** | ⚠️ | `daqr/algorithms/` or embedded | Check Algorithm 3 |
| **QBGP Protocol** | ⚠️ | `daqr/core/networkenvironment.py` | Test route propagation |
| **Physics Models** | ✅ | `quantum_physics.py` | Run fidelity test |
| **Configuration** | ✅ | `ExperimentConfiguration['paper7']` | Match Liu et al. params |
| **Validation Tests** | ❌ | Need to create | Run 3 validation scripts |

## Quick Diagnosis (< 5 min)

```bash
# 1. Create diagnostic script
cat > scripts/diagnostic_paper7.py << 'EOF'
# [See PAPER7_VALIDATION.md Part 1]
EOF

# 2. Run it
python scripts/diagnostic_paper7.py

# 3. Check for ✗ marks
# All ✓ or ⚠ = Proceed to validation tests
# Any ✗ = Fix before continuing
```

## Three Validation Tests

### Test 1: Route Propagation (5-10 min)
- **What**: QBGP route announcements propagate across ASes
- **Expected**: <500ms convergence, >95% reachability
- **Figure**: Liu et al. Figure 4
- **File**: `scripts/validation_test_1_convergence.py`
- **Run**: `python scripts/validation_test_1_convergence.py`

### Test 2: QPSEL Bounces (3-5 min)
- **What**: Online top-K uses fewer bounces than vanilla
- **Expected**: 2-3x more efficient than vanilla benchmarking
- **Figure**: Liu et al. Figures 6-7
- **File**: `scripts/validation_test_2_bounces.py`
- **Run**: `python scripts/validation_test_2_bounces.py`

### Test 3: Goodput (3-5 min)
- **What**: Multipath improves performance under noise
- **Expected**: >15% improvement under high noise
- **Figure**: Liu et al. Figures 8-14
- **File**: `scripts/validation_test_3_goodput.py`
- **Run**: `python scripts/validation_test_3_goodput.py`

## Integration Checklist

**Before saying "Paper 7 is integrated":**

- [ ] Diagnostic script ✓ PASSES (all checks pass/warn, nothing fails)
- [ ] Test 1 ✓ PASSES (convergence <500ms, reachability >95%)
- [ ] Test 2 ✓ PASSES (QPSEL > Pure Expl > Vanilla)
- [ ] Test 3 ✓ PASSES (improvement >15% for high noise)
- [ ] Paper 2 experiments ✓ STILL WORK (no regression)
- [ ] Results saved ✓ YES (JSON files in results/)
- [ ] Documentation ✓ YES (README updated)

## Key Paper References

**Liu et al. 2024 QBGP - Key Sections:**
- **Algorithm 1**: Network Benchmarking
- **Algorithm 2**: QBGP Route Propagation
- **Algorithm 3**: QPSEL Online Top-K Path Selection
- **Algorithm 4**: QBGP Route Propagation (loop prevention)
- **Figure 4**: Route propagation convergence
- **Figures 6-7**: QPSEL bounce efficiency
- **Figures 8-14**: Goodput under noise
- **Theorem 1**: QPSEL identifies top-K with prob 1-δ
- **Theorem 2**: Sample complexity O(Σ Δ_i^(-2) log(L/Δ_i))

## Expected Results Summary

### Figure 4: Route Convergence
```
N_ASes | Conv (ms) | Reachability
30     | ~50       | ~98%
50     | ~125      | ~97%
80     | ~300      | ~95%
```
✓ Linear or sublinear scaling

### Figures 6-7: QPSEL Efficiency
```
L=8 paths | Vanilla | Pure Expl | QPSEL | Ratio
Config 1  | 160     | 200       | 67    | 2.4x
Config 2  | 160     | 300       | 100   | 1.6x
```
✓ Consistent 2-3x improvement

### Figures 8-14: Goodput
```
Noise  | Baseline | QPSEL | Improvement
10%    | 0.729    | 0.829 | 13.7%
50%    | 0.129    | 0.198 | 53.5%
80%    | 0.022    | 0.042 | 93.5%
```
✓ Improvement increases with noise

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: Paper7ASTopologyGenerator` | Class not found | Add to topology_generator.py |
| Context ranges out of bounds | Wrong formulas | Verify hop=(len-1), avg_degree=mean(degrees), length=sum(edges) |
| QPSEL bounces not 3x better | Info gain wrong | Check I(p,m) = 16A²p^(4m²)/(2m) |
| Goodput <15% improvement | Load balancing missing | Verify round-robin across top-K paths |
| Convergence >500ms | Algorithm slow | Use BFS/DFS with queue, not recursive |

## Next Steps

1. **Now**: Run diagnostic script → check all ✓ or ⚠
2. **Then**: Run Test 1 → verify convergence
3. **Then**: Run Test 2 → verify bounce efficiency
4. **Then**: Run Test 3 → verify goodput
5. **Then**: Collect results → write paper section
6. **Finally**: Submit → "We validated against Paper 7 testbed"

## Success Criteria

✅ **Reproducibility**: Error bars < 10%  
✅ **Figures**: All 4, 6-7, 8-14 reproducible  
✅ **Integration**: Paper 2 + Paper 7 both work  
✅ **Documentation**: README explains integration  
✅ **Ready for Publication**: All checks pass  

---

**Estimated time to full validation**: 30-60 min (diagnostic + 3 tests + collection)

**Questions?** Check PAPER7_VALIDATION.md for detailed explanations of each test.
