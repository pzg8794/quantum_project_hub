# Quantum Testbeds - Master Quick Reference

**Framework**: H-MABs Evaluation Framework (Multi-Armed Bandits for Quantum Networks)  
**Last Updated**: January 30, 2026  
**Status**: ✅ All three testbeds integrated and organized

---

## Overview

This project integrates three distinct quantum routing research papers into a unified evaluation framework:

1. **Paper 2** (Chaudhary et al. 2023): MAB-based quantum network routing with entanglement swapping
2. **Paper 7** (Liu et al. 2024): QBGP - Quantum BGP with online path selection
3. **Paper 12** (Wang et al. 2024): QuARC - Qubit allocation with fusion gates

---

## Quick Navigation

### 🔗 Master Documents

| Testbed | Quick Ref | Location | Status |
|---------|-----------|----------|--------|
| **Paper 2** | [QUICK_REFERENCE.md](../../Testbeds/Paper2-Quantum-Network-MultiArmedBandits-main/QUICK_REFERENCE.md) | `Testbeds/Paper2-...` | ✅ Integrated |
| **Paper 7** | [QUICK_REFERENCE.md](../../Testbeds/Paper7-lizhuohua-quantum-bgp-online-path-selection-aeb35c0/QUICK_REFERENCE.md) | `Testbeds/Paper7-...` | ✅ Integrated |
| **Paper 12** | [Paper12_Quick_Reference.md](../../hybrid_variable_framework/docs/testbeds/Paper12_Quick_Reference.md) | `hybrid_variable_framework/docs/testbeds/` | ✅ Integrated |

---

## Paper Comparison Matrix

### Research Focus
| Aspect | Paper 2 | Paper 7 | Paper 12 |
|--------|---------|---------|----------|
| **Title** | Quantum Network MAB | QBGP Routing | QuARC Allocation |
| **Authors** | Chaudhary et al. 2023 | Liu et al. 2024 | Wang et al. 2024 |
| **Primary Algorithm** | UCB (Multi-Armed Bandits) | BGP-inspired Protocol | Qubit Allocation |
| **Quantum Focus** | Entanglement swapping | Delay-aware routing | Fusion gate optimization |

### Implementation Details
| Aspect | Paper 2 | Paper 7 | Paper 12 |
|--------|---------|---------|----------|
| **Language** | MATLAB | Python | Python |
| **Test Framework** | Manual testing | Unit tests | Unit tests (6 tests) |
| **Network Sizes** | Up to 200 nodes | 50-400 nodes | 100 nodes (baseline) |
| **Quantum Model** | Fidelity-based | Delay-aware | Fusion success |
| **Testing Status** | ✅ Documented | ✅ Automated | ✅ Automated |

### Baseline Parameters
| Parameter | Paper 2 | Paper 7 | Paper 12 |
|-----------|---------|---------|----------|
| **Network Nodes** | 20 (scalable) | 100 | 100 |
| **Avg Degree** | 4 | 6 | 6 |
| **S-D Pairs** | Variable | 4 | 10 |
| **Key Success Rate** | E_p: 0.7, q: 0.9 | Fidelity: ≥0.85 | Fusion: 0.9, Entangle: 0.6 (54%) |

---

## File Organization by Testbed

### Paper 2 Directory Structure
```
Testbeds/Paper2-Quantum-Network-MultiArmedBandits-main/
├── QUICK_REFERENCE.md               ← START HERE
├── Paper2_Integration_Report.md      ← Integration details
├── Paper2_Integration_Checklist.txt  ← Verification status
├── Paper2_Test_Commands.md           ← Test procedures
├── MAB_UCB_QNetwork_Routing.m        ← Main algorithm (MATLAB)
├── [other MATLAB implementation files]
├── paper2_chaudhary2023quantum.pdf   ← Research paper
└── paper2_framework.png              ← Architecture diagram
```

### Paper 7 Directory Structure
```
Testbeds/Paper7-lizhuohua-quantum-bgp-online-path-selection-aeb35c0/
├── QUICK_REFERENCE.md               ← START HERE
├── validation/
│   ├── PAPER7_QUICKREF.md            ← Original quick ref
│   ├── PAPER7_VALIDATION.md          ← Validation guide
│   ├── PAPER7_STATUS_REPORT.md       ← Status findings
│   ├── PAPER7_SUMMARY.md             ← Comprehensive summary
│   └── [generated charts and plots]
├── main.py                           ← Main QBGP implementation
├── [other Python implementation files]
├── topology_data/                    ← Network topologies
├── network_benchmarking/             ← Benchmarking utilities
├── paper7_liu2024qbgp.pdf            ← Research paper
└── README.md                         ← Project overview
```

### Paper 12 Directory Structure
```
hybrid_variable_framework/Dynamic_Routing_Eval_Framework/
├── QUICK_REFERENCE.md               ← START HERE (UPDATED)
├── INDEX.md                          ← Complete documentation index
├── README_TESTING.md                 ← Testing procedures
├── DELIVERY_SUMMARY.md               ← Delivery scope
│
├── run_paper12_sanity_tests.py       ← Unit test suite (23 KB)
├── run_paper7_sanity_tests.py        ← Paper 7 unit tests
├── run_tests.sh                      ← Test runner script
│
├── PAPER12_TESTS_README.md           ← Full test documentation
├── PAPER12_TESTING_SUMMARY.md        ← Workflow overview
├── PAPER7_vs_PAPER12_TESTING.md      ← Testing comparison
│
├── results/
│   ├── paper12_sanity_tests.json     ← Paper 12 results
│   └── paper7_sanity_tests.json      ← Paper 7 results
│
└── notebooks/
    └── H-MABs_Eval-T_XQubit_Alloc_XQRuns copy.ipynb
```

### Root Level Documentation
```
Root/PAPER12_* files
├── PAPER12_ALLOCATOR_EXECUTION.md           ← Allocator execution flow
├── PAPER12_PARAMETERS_VALIDATION.md         ← Parameter validation
├── PAPER12_DOCUMENTATION_UPDATE_COMPLETE.md ← Update tracking
├── PAPER12_UPDATES_CHECKLIST.md             ← Updates made
├── ASSESSMENT_CORRECTION_SUMMARY.md         ← Assessment validation
└── BASELINE_ASSESSMENT_QUICK_REF.md         ← Baseline reference
```

---

## Getting Started by Testbed

### Paper 2 (MATLAB-based)
```bash
# 1. Navigate to testbed
cd Testbeds/Paper2-Quantum-Network-MultiArmedBandits-main/

# 2. Read quick reference
cat QUICK_REFERENCE.md

# 3. Review integration status
cat Paper2_Integration_Report.md

# 4. Check test commands
cat Paper2_Test_Commands.md
```

### Paper 7 (Python implementation)
```bash
# 1. Navigate to testbed
cd Testbeds/Paper7-lizhuohua-quantum-bgp-online-path-selection-aeb35c0/

# 2. Read quick reference
cat QUICK_REFERENCE.md

# 3. Review validation
cat validation/PAPER7_SUMMARY.md

# 4. Run unit tests
cd ../../hybrid_variable_framework/Dynamic_Routing_Eval_Framework/
python run_paper7_sanity_tests.py
```

### Paper 12 (Full test suite - RECOMMENDED)
```bash
# 1. Navigate to test framework
cd hybrid_variable_framework/Dynamic_Routing_Eval_Framework/

# 2. Read quick reference
cat QUICK_REFERENCE.md

# 3. Run unit tests (0.1 seconds)
python run_paper12_sanity_tests.py

# 4. Check results
cat results/paper12_sanity_tests.json | python -m json.tool

# 5. Review complete documentation
cat PAPER12_TESTS_README.md
```

---

## Testing Summary

### Unit Test Coverage
| Paper | Test File | Tests | Status | Command |
|-------|-----------|-------|--------|---------|
| 12 | `run_paper12_sanity_tests.py` | 6 tests | ✅ All passing | `python run_paper12_sanity_tests.py` |
| 7 | `run_paper7_sanity_tests.py` | 5+ tests | ✅ Available | `python run_paper7_sanity_tests.py` |
| 2 | Manual tests | Per docs | ✅ Documented | See `Paper2_Test_Commands.md` |

### Test Results Location
```
Dynamic_Routing_Eval_Framework/
└── results/
    ├── paper12_sanity_tests.json      (Auto-generated after test run)
    └── paper7_sanity_tests.json       (Auto-generated after test run)
```

---

## Key Parameters by Testbed

### Paper 2 (Baseline)
```
Entanglement Probability (E_p): 0.7
Quantum Gate Fidelity (q): 0.9
Network Nodes: 20 (scalable to 200)
Topology: Standard graph
```

### Paper 7 (Baseline)
```
Network Nodes: 100
Avg Degree: 6 (Waxman topology)
Min Fidelity: 0.85
QoS Threshold: 0.80
S-D Pairs: 4 (in tests)
```

### Paper 12 (Baseline - OFFICIAL)
```
Fusion Probability (q): 0.9
Entanglement Probability (E_p): 0.6
Combined Success Rate: 54% (0.9 × 0.6) ✅
Network Nodes: 100
Avg Degree: 6 (Waxman topology)
S-D Pairs: 10
Total Timeslots: 5000
```

---

## Documentation Hierarchy

### Quick References (Start here - 5 min read)
- `Testbeds/Paper2-.../QUICK_REFERENCE.md`
- `Testbeds/Paper7-.../QUICK_REFERENCE.md`
- `Dynamic_Routing_Eval_Framework/QUICK_REFERENCE.md`

### Comprehensive Guides (10-15 min read)
- Paper 2: `Paper2_Integration_Report.md`
- Paper 7: `validation/PAPER7_SUMMARY.md`
- Paper 12: `PAPER12_TESTS_README.md` + `PAPER12_TESTING_SUMMARY.md`

### Implementation Details (Reference)
- Paper 2: Individual MATLAB files
- Paper 7: `main.py`, `protocols.py`, `components.py`
- Paper 12: `run_paper12_sanity_tests.py` source code

### Testing Procedures
- Paper 2: `Paper2_Test_Commands.md`
- Paper 7: `validation/PAPER7_VALIDATION.md`
- Paper 12: `README_TESTING.md` + automated unit tests

---

## Framework Integration Status

### Paper 2 ✅
- [x] Codebase integrated
- [x] Documentation organized
- [x] Integration verified
- [x] Quick reference created
- [ ] Python translation (future)
- [ ] Unit tests (future)

### Paper 7 ✅
- [x] Codebase integrated
- [x] Documentation organized
- [x] Python implementation available
- [x] Unit test framework created
- [x] Validation procedures documented
- [x] Quick reference created

### Paper 12 ✅
- [x] Codebase integrated
- [x] Comprehensive testing framework
- [x] 6 unit tests (all passing)
- [x] Full documentation suite
- [x] Parameters validated
- [x] Quick reference created
- [x] Allocator execution documented
- [x] Baseline parameters clarified

---

## Key Resources by Need

### "I want to understand what each paper does"
→ Read Master Documents:
- `QUICK_REFERENCE.md` in each testbed folder

### "I want to run tests quickly"
→ Paper 12 recommended:
```bash
cd hybrid_variable_framework/Dynamic_Routing_Eval_Framework/
python run_paper12_sanity_tests.py
```

### "I want to compare testing strategies"
→ Read:
- `Dynamic_Routing_Eval_Framework/PAPER7_vs_PAPER12_TESTING.md`

### "I need to understand parameters"
→ Check:
- Paper 2: `Paper2_Integration_Report.md`
- Paper 7: `validation/PAPER7_SUMMARY.md`
- Paper 12: `PAPER12_PARAMETERS_VALIDATION.md`

### "I want complete implementation details"
→ Review:
- Paper 2: MATLAB source code
- Paper 7: `main.py`, `protocols.py`
- Paper 12: `run_paper12_sanity_tests.py`

---

## Quick Command Reference

### Run All Paper 12 Tests
```bash
cd hybrid_variable_framework/Dynamic_Routing_Eval_Framework/
python run_paper12_sanity_tests.py
```

### Run All Paper 7 Tests
```bash
cd hybrid_variable_framework/Dynamic_Routing_Eval_Framework/
python run_paper7_sanity_tests.py
```

### View Paper 12 Test Results
```bash
cat hybrid_variable_framework/Dynamic_Routing_Eval_Framework/results/paper12_sanity_tests.json | python -m json.tool
```

### Read Any Quick Reference
```bash
cat Testbeds/Paper2-...QUICK_REFERENCE.md
cat Testbeds/Paper7-.../QUICK_REFERENCE.md
cat hybrid_variable_framework/Dynamic_Routing_Eval_Framework/QUICK_REFERENCE.md
```

---

## Project Structure Summary

```
GA-Work/
│
├── Testbeds/
│   ├── Paper2-Quantum-Network-MultiArmedBandits-main/
│   │   └── QUICK_REFERENCE.md      ← Paper 2 Quick Ref
│   │
│   └── Paper7-lizhuohua-quantum-bgp-online-path-selection-aeb35c0/
│       └── QUICK_REFERENCE.md      ← Paper 7 Quick Ref
│
├── hybrid_variable_framework/
│   └── Dynamic_Routing_Eval_Framework/
│       ├── QUICK_REFERENCE.md      ← Paper 12 Quick Ref (UPDATED)
│       ├── run_paper12_sanity_tests.py
│       ├── run_paper7_sanity_tests.py
│       ├── PAPER12_TESTS_README.md
│       ├── PAPER7_vs_PAPER12_TESTING.md
│       └── results/
│           ├── paper12_sanity_tests.json
│           └── paper7_sanity_tests.json
│
└── Root-Level Documentation
    ├── PAPER12_ALLOCATOR_EXECUTION.md
    ├── PAPER12_PARAMETERS_VALIDATION.md
    ├── PAPER12_DOCUMENTATION_UPDATE_COMPLETE.md
    ├── PAPER12_UPDATES_CHECKLIST.md
    ├── ASSESSMENT_CORRECTION_SUMMARY.md
    └── BASELINE_ASSESSMENT_QUICK_REF.md
```

---

## Next Steps

1. **Choose a testbed** based on your needs
2. **Read the QUICK_REFERENCE.md** in that testbed (5 min)
3. **Review associated documentation** for details (10-15 min)
4. **Run unit tests** if applicable (30 sec - 2 min)
5. **Check results** in results/ folder

---

## Support & References

### Paper Sources
- **Paper 2**: `Testbeds/Paper2-.../paper2_chaudhary2023quantum.pdf`
- **Paper 7**: `Testbeds/Paper7-.../paper7_liu2024qbgp.pdf`
- **Paper 12**: Referenced in Paper 12 documentation

### Key Documentation Files
- All `QUICK_REFERENCE.md` files (one per testbed)
- All `README.md` files in respective testbeds
- Validation and testing guides in each testbed

### Framework Documentation
- `hybrid_variable_framework/README.md`
- `hybrid_variable_framework/Dynamic_Routing_Eval_Framework/INDEX.md`
- Root-level PAPER12_* documentation files

---

**Status**: ✅ **ALL TESTBEDS ORGANIZED AND DOCUMENTED**

Last organized: January 30, 2026
