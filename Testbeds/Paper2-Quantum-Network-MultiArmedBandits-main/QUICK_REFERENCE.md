# Paper 2 - Quick Reference Guide

**Paper**: Chaudhary et al. 2023 - Quantum Network Multi-Armed Bandits  
**Location**: `Testbeds/Paper2-Quantum-Network-MultiArmedBandits-main/`  
**Status**: ✅ Integrated testbed

---

## TL;DR

Paper 2 (Chaudhary et al. 2023) implements multi-armed bandits for quantum network routing. MATLAB implementation with entanglement swapping, fidelity calculations, and QoS-aware path selection.

**Key Focus**: Quantum channel fidelity and entanglement-aware network optimization.

---

## All Paper 2 Associated Files

### Core Implementation Files (MATLAB)
| File | Purpose |
|------|---------|
| `MAB_UCB_QNetwork_Routing.m` | Main MAB implementation for quantum routing |
| `QNetworkGraph_LearningAlgo.m` | Q-learning network graph algorithms |
| `EntanglementSwap_Noises.m` | Entanglement swapping with noise models |
| `EntanglementSwap_NoiseProbabilities.m` | Noise probability calculations |
| `Fidelity_FeasiblePaths.m` | Fidelity computation for feasible paths |
| `Link_EntangledPair_Fidelity.m` | Per-link entanglement pair fidelity |
| `CascadedFidelity.m` | Cascaded fidelity across path |
| `FiberLoss.m` | Quantum fiber loss modeling |
| `dijkstra_QNetwork.m` | Dijkstra's algorithm for quantum networks |
| `kShortestPath_QNetwork.m` | K-shortest paths for quantum networks |
| `getProjectors.m` | Quantum state projectors |
| `operationAfterMeasure_ES_GateErr.m` | Post-measurement operations with gate errors |

### Documentation Files
| File | Purpose |
|------|---------|
| `Paper2_Quick_Reference.md` | Quick reference (original) |
| `Paper2_Integration_Report.md` | Integration analysis report |
| `Paper2_Integration_Checklist.txt` | Integration verification checklist |
| `Paper2_Test_Commands.md` | Test execution commands |
| `README.md` | Project overview |

### Reference Materials
| File | Purpose |
|------|---------|
| `paper2_chaudhary2023quantum.pdf` | Original research paper |
| `paper2_framework.png` | Framework architecture diagram |

---

## Paper 2 Parameters

### Quantum Network Configuration
```matlab
% Entanglement parameters
E_p = 0.7           % Entanglement success probability
q = 0.9             % Quantum gate fidelity

% Network topology
n_nodes = 20        % Network size (scalable)
avg_degree = 4      % Average node connectivity

% MAB parameters
epsilon = 0.1       % Exploration rate (ε-greedy)
confidence = 0.95   % UCB confidence level
```

### Core Features
- ✅ **Entanglement Swapping**: Noise-aware model
- ✅ **Fidelity Calculation**: Path-level and link-level
- ✅ **MAB Learning**: UCB algorithm for path selection
- ✅ **QoS Metrics**: Fidelity as reward signal
- ✅ **Scalability**: Tested up to 200 nodes

---

## Key Differences from Other Papers

| Aspect | Paper 2 | Paper 7 | Paper 12 |
|--------|---------|---------|----------|
| **Language** | MATLAB | Python | Python |
| **Algorithm** | UCB (MAB) | QBGP protocol | QuARC (Fusion gates) |
| **Focus** | Entanglement swapping | BGP-inspired routing | Qubit allocation |
| **Network Size** | Up to 200 nodes | 50-400 nodes | 100 nodes |
| **Quantum Model** | Fidelity-based | Delay-aware | Fusion success |
| **MAB Strategy** | UCB | Path scoring | Arm allocation |

---

## Running Paper 2 Experiments

### Basic Setup
```bash
cd Testbeds/Paper2-Quantum-Network-MultiArmedBandits-main/
```

### MATLAB Execution (in MATLAB environment)
```matlab
% Load and run main routing algorithm
run MAB_UCB_QNetwork_Routing.m

% Or run individual tests
run Paper2_Test_Commands.md
```

### Test Execution
See `Paper2_Test_Commands.md` for specific test procedures.

---

## Integration Notes

### With H-MABs Framework
- **Status**: ✅ Integrated
- **Integration Report**: `Paper2_Integration_Report.md`
- **Verification**: `Paper2_Integration_Checklist.txt`

### MATLAB to Python Bridge
- MATLAB code serves as reference implementation
- Python quantum physics modules can reference Paper 2 concepts
- Fidelity calculations implemented in Python counterparts

---

## File Organization

```
Paper2-Quantum-Network-MultiArmedBandits-main/
├── Core Implementation (MATLAB files)
│   ├── MAB_UCB_QNetwork_Routing.m          ← Main algorithm
│   ├── EntanglementSwap_*.m                ← Entanglement models
│   ├── Fidelity_*.m                        ← Fidelity calculations
│   ├── dijkstra_QNetwork.m                 ← Path algorithms
│   └── ...other utility functions
│
├── Documentation
│   ├── Paper2_Quick_Reference.md           ← Quick reference
│   ├── Paper2_Integration_Report.md        ← Integration analysis
│   ├── Paper2_Integration_Checklist.txt    ← Verification checklist
│   ├── Paper2_Test_Commands.md             ← Test procedures
│   └── README.md
│
├── Reference Materials
│   ├── paper2_chaudhary2023quantum.pdf     ← Research paper
│   └── paper2_framework.png                ← Architecture diagram
```

---

## Quick Links

- **Research Paper**: `paper2_chaudhary2023quantum.pdf`
- **Integration Report**: [Paper2_Integration_Report.md](Paper2_Integration_Report.md)
- **Test Guide**: [Paper2_Test_Commands.md](Paper2_Test_Commands.md)
- **Main Algorithm**: [MAB_UCB_QNetwork_Routing.m](MAB_UCB_QNetwork_Routing.m)

---

## Key Takeaways

1. **MAB Learning**: Uses Upper Confidence Bound (UCB) algorithm for path selection
2. **Quantum Realism**: Includes entanglement swapping and fidelity degradation
3. **Scalability**: Tested on networks up to 200 nodes
4. **Reference**: Provides baseline for quantum network routing comparisons
5. **Integration**: Concepts integrated into H-MABs framework for Python implementation

---

## Next Steps

1. **Review** `Paper2_Integration_Report.md` for integration details
2. **Check** `Paper2_Integration_Checklist.txt` for verification status
3. **Run** experiments following `Paper2_Test_Commands.md`
4. **Compare** with Paper 7 and Paper 12 results in framework

---

**Last Updated**: January 30, 2026  
**Status**: ✅ Complete Integration
