# Paper 7 - Quick Reference Guide

**Paper**: Liu et al. 2024 QBGP - Quantum BGP Online Path Selection  
**Location**: `Testbeds/Paper7-lizhuohua-quantum-bgp-online-path-selection-aeb35c0/`  
**Status**: ✅ Integrated testbed with Python implementation

---

## TL;DR

Paper 7 (Liu et al. 2024) implements QBGP - a BGP-inspired quantum routing protocol with online path selection. Python implementation with network benchmarking, delay-aware routing, and quantum fidelity modeling.

**Quick Start**: See `validation/` folder for testing framework and quick reference.

---

## All Paper 7 Associated Files

### Core Implementation Files (Python)
| File | Purpose |
|------|---------|
| `main.py` | Main QBGP protocol implementation |
| `components.py` | Protocol components and modules |
| `protocols.py` | Protocol definitions and logic |
| `event_generators.py` | Event generation for simulation |
| `packets.py` | Quantum packet definitions |
| `import_data.py` | Network data import utilities |
| `utils.py` | Utility functions |
| `__init__.py` | Package initialization |

### Testing & Validation Framework
| File | Location | Purpose |
|------|----------|---------|
| `run_paper7_sanity_tests.py` | `Dynamic_Routing_Eval_Framework/` | Unit test suite (5+ tests) |
| `PAPER7_vs_PAPER12_TESTING.md` | `Dynamic_Routing_Eval_Framework/` | Testing strategy comparison |
| `PAPER7_QUICKREF.md` | `validation/` | Original quick reference |
| `PAPER7_VALIDATION.md` | `validation/` | Validation procedures |
| `PAPER7_STATUS_REPORT.md` | `validation/` | Status and findings report |
| `PAPER7_SUMMARY.md` | `validation/` | Comprehensive summary |

### Network Data & Topology
| Directory | Purpose |
|-----------|---------|
| `topology_data/` | Network topology datasets |
| `network_benchmarking/` | Network benchmarking utilities |

### Reference Materials
| File | Purpose |
|------|---------|
| `paper7_liu2024qbgp.pdf` | Original research paper |
| `README.md` | Project overview |
| `LICENSE` | Project license |
| `chaudhary2023quantum.zip` | Related paper 2 archive |

---

## Paper 7 Configuration

### Network Parameters
```python
# Standard topology configuration
n_nodes = 100           # Network size
avg_degree = 6          # Average node connectivity (Waxman topology)
num_paths = 4           # Source-destination pairs
delay_model = 'linear'  # Delay calculation model

# Quantum parameters
fidelity_min = 0.85     # Minimum acceptable fidelity
fidelity_decay = 0.95   # Per-hop fidelity decay
qos_threshold = 0.80    # QoS acceptance threshold
```

### QBGP Protocol Features
- ✅ **Online Path Selection**: Real-time path optimization
- ✅ **Delay-Aware Routing**: Incorporates network delays
- ✅ **Fidelity Modeling**: Quantum fidelity awareness
- ✅ **Scalability**: Tested 50-400 nodes
- ✅ **Comparative Analysis**: BGP inspiration with quantum enhancements

---

## Testing Framework

### Available Tests (Unit Tests)
- Test 1: Topology + paths
- Test 2: Context feature validation
- Test 3: Reward range validation
- Test 4: Physics parameters (Paper 7 specific)
- Test 5: Integration format check

### Running Tests
```bash
cd hybrid_variable_framework/Dynamic_Routing_Eval_Framework

# Run Paper 7 unit tests
python run_paper7_sanity_tests.py

# View results
cat results/paper7_sanity_tests.json | python -m json.tool
```

---

## File Organization

```
Paper7-lizhuohua-quantum-bgp-online-path-selection-aeb35c0/
│
├── Core Implementation (Python)
│   ├── main.py                          ← Main QBGP protocol
│   ├── components.py                    ← Protocol components
│   ├── protocols.py                     ← Protocol definitions
│   ├── event_generators.py              ← Simulation events
│   ├── packets.py                       ← Quantum packets
│   ├── import_data.py                   ← Data utilities
│   ├── utils.py                         ← Helper functions
│   └── __init__.py                      ← Package init
│
├── Testing & Validation
│   ├── validation/
│   │   ├── PAPER7_QUICKREF.md           ← Quick reference
│   │   ├── PAPER7_VALIDATION.md         ← Validation guide
│   │   ├── PAPER7_STATUS_REPORT.md      ← Status report
│   │   ├── PAPER7_SUMMARY.md            ← Full summary
│   │   ├── chart.png                    ← Results visualization
│   │   ├── chart2.png                   ← Additional charts
│   │   └── generated-image.png          ← Generated plots
│   │
│   └── (In Dynamic_Routing_Eval_Framework/)
│       ├── run_paper7_sanity_tests.py   ← Unit tests
│       ├── PAPER7_vs_PAPER12_TESTING.md ← Testing comparison
│       └── results/paper7_sanity_tests.json
│
├── Network Data
│   ├── topology_data/                   ← Network topologies
│   └── network_benchmarking/            ← Benchmarking tools
│
├── Documentation
│   ├── README.md                         ← Project overview
│   ├── LICENSE                           ← Project license
│   └── plots/                            ← Result plots
│
└── Reference Materials
    ├── paper7_liu2024qbgp.pdf           ← Research paper
    ├── chaudhary2023quantum.zip         ← Paper 2 reference
    └── Release of QBGP paper codebase.zip
```

---

## Key Features Comparison

| Feature | Paper 7 QBGP | Paper 12 QuARC |
|---------|--------------|----------------|
| **Protocol** | BGP-inspired | Fusion gate based |
| **Routing** | Online path selection | Qubit allocation |
| **Delay** | Explicitly modeled | Not primary focus |
| **Fidelity** | Per-path fidelity | Fusion success rate |
| **Learning** | Path scoring | MAB algorithms |
| **Scalability** | 50-400 nodes | 100 nodes (baseline) |
| **Implementation** | Python | Python |

---

## Integration with H-MABs Framework

### Python Implementation
- ✅ Full Python translation from original codebase
- ✅ Integration with H-MABs testbed
- ✅ Unit testing framework (parity with Paper 12)
- ✅ Results comparison and analysis

### Key Integration Points
1. **Topology Generation**: Waxman model (consistent with other papers)
2. **Context Features**: [hop_count, normalized_degree, fidelity_prob]
3. **Rewards**: Fidelity-based rewards [5, 100] range
4. **Testing**: Unified test suite approach

---

## Quick Links

- **Research Paper**: `paper7_liu2024qbgp.pdf`
- **Quick Reference** (Original): `validation/PAPER7_QUICKREF.md`
- **Validation Guide**: `validation/PAPER7_VALIDATION.md`
- **Status Report**: `validation/PAPER7_STATUS_REPORT.md`
- **Full Summary**: `validation/PAPER7_SUMMARY.md`
- **Unit Tests**: `../Dynamic_Routing_Eval_Framework/run_paper7_sanity_tests.py`
- **Test Comparison**: `../Dynamic_Routing_Eval_Framework/PAPER7_vs_PAPER12_TESTING.md`

---

## Testing Strategy

### Unit Tests (Recommended)
```bash
# Fast validation (0.1 seconds)
python run_paper7_sanity_tests.py

# View specific results
python run_paper7_sanity_tests.py | grep "TEST"
```

### Manual Testing
Follow procedures in `validation/PAPER7_VALIDATION.md`

### Benchmark Testing
Use `network_benchmarking/` utilities for performance evaluation

---

## Key Parameters Validated

```python
# Paper 7 QBGP standard configuration
topology_nodes = 100           # Standard test size
avg_degree = 6                 # Waxman topology
fidelity_min = 0.85            # Minimum fidelity threshold
qos_threshold = 0.80           # QoS acceptance level
delay_model = 'linear'         # Per-hop delay calculation
```

---

## Next Steps

1. **Review** `validation/PAPER7_SUMMARY.md` for overview
2. **Check** `validation/PAPER7_VALIDATION.md` for testing procedures
3. **Run** `python run_paper7_sanity_tests.py` for unit tests
4. **Compare** with Paper 12 using `PAPER7_vs_PAPER12_TESTING.md`
5. **Analyze** results in `validation/` folder

---

## Key Takeaways

1. **QBGP Protocol**: BGP-inspired quantum routing with online path selection
2. **Delay-Aware**: Explicitly models network delays in routing decisions
3. **Fidelity Focus**: Uses quantum fidelity as primary optimization metric
4. **Scalable**: Tested on networks from 50 to 400 nodes
5. **Python Native**: Fully implemented in Python for H-MABs integration
6. **Well-Documented**: Comprehensive validation and testing framework

---

**Last Updated**: January 30, 2026  
**Status**: ✅ Complete Integration with Testing Framework  
**Test Coverage**: 5+ unit tests with JSON result tracking
