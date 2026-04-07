# UDRM Integration Quick Reference
**Quick Guide for Tracking & Executing UDRM Integration**  
**Status:** Active Planning Phase  
**Print/Bookmark This**

---

## 🎯 At a Glance

**UDRM Integration = 8 Integration Points**

```
START HERE ↓
┌─────────────────────────────────────┐
│ P8: Config (30 min)                 │ ← Do FIRST (unblocks everything)
│ P6: UDRM Class (2-3 hrs)            │ ← Do SECOND (needed by all)
└─────────────┬───────────────────────┘
              ↓
    ┌─────────────────────┐
    │ Framework Level     │
    │ P1: Entry (15 min)  │
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │ Hierarchy Flow      │
    │ P2: Eval (20 min)   │
    │ P3: Exp (20 min)    │
    │ P4: Run (30 min)    │
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │ Integration         │
    │ P7: Unc (45 min)    │
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │ Cleanup             │
    │ P5: Cleanup (30 min)│
    └─────────────────────┘
```

---

## 📋 Integration Points Checklist

### Setup Phase (Do First)
- [ ] **P8** - Add UDRM config parameters
- [ ] **P6** - Implement UDRMObject class

### Integration Phase (Do After Setup)
- [ ] **P1** - Add UDRM to framework init
- [ ] **P2** - Create UDRM in evaluator loop
- [ ] **P3** - Pass UDRM through experiment loop
- [ ] **P4** - Call process_transition in run loop
- [ ] **P7** - Integrate uncertainty extraction
- [ ] **P5** - Add cleanup aggregation

### Testing Phase
- [ ] Unit test UDRM class
- [ ] Integration test full flow
- [ ] Validate against paper Algorithm 1

---

## 🔑 Key Code Patterns

### Pattern 1: Creating & Initializing
```python
# P1: Framework init
self.udrm_obj = None
self.udrm_config = {...}

# P2: Evaluator loop
self.udrm_obj = self.create_udrm(physics_model)
self.udrm_obj.initialize_exploration_phase(num_timesteps)
```

### Pattern 2: Passing Through Hierarchy
```python
# P3: Experiment loop
self.run_experiments_for_model(..., udrm=self.udrm_obj, ...)

# P4: Run loop
udrm.process_transition(state, action, reward, next_state, timestep, ...)
```

### Pattern 3: Tracking State
```python
# P2: Store tracking dict
evaluator_udrm_states = {
    model: {'udrm_obj': obj, 'alpha_history': [], ...}
}

# P3: Update history after each run
udrm_state_tracker['alpha_history'].append(udrm.alpha)

# P5: Aggregate at cleanup
summary = udrm_obj.get_state_summary()
```

---

## 📍 File Locations

| Point | File | Method | Action |
|-------|------|--------|--------|
| **P1** | runner.py | `__init__` | Add 2 lines |
| **P2** | runner.py | `run_evaluator` | Add 10 lines |
| **P3** | runner.py | `run_experiments` | Add 5 lines |
| **P4** | runner.py | `run_single_experiment` | Add 1 call |
| **P5** | runner.py | `cleanup_evaluator` | New method |
| **P6** | udrm.py | (new file) | ~200 lines |
| **P7** | algorithms | methods | Extract uncertainty |
| **P8** | config | YAML/dict | Add 6 params |

---

## 💡 Remember

1. **Order Matters** - Start with P8 & P6, they unblock others
2. **Follow Allocator Pattern** - UDRM mirrors allocator creation/passing
3. **State is King** - UDRM isn't configuration, it's a stateful object
4. **Persist Through Experiments** - Alpha/beta DON'T reset between runs
5. **Pass Down, Aggregate Up** - Created at top, flows down, aggregates at cleanup

---

## ⚡ Quick Status Check

Update this when you start each point:

```
Session Date: _______________

P8 ☐ Not Started  ☐ In Progress  ☐ Complete  (Time: __)
P6 ☐ Not Started  ☐ In Progress  ☐ Complete  (Time: __)
P1 ☐ Not Started  ☐ In Progress  ☐ Complete  (Time: __)
P2 ☐ Not Started  ☐ In Progress  ☐ Complete  (Time: __)
P3 ☐ Not Started  ☐ In Progress  ☐ Complete  (Time: __)
P4 ☐ Not Started  ☐ In Progress  ☐ Complete  (Time: __)
P7 ☐ Not Started  ☐ In Progress  ☐ Complete  (Time: __)
P5 ☐ Not Started  ☐ In Progress  ☐ Complete  (Time: __)

Overall: ___% Complete
Blockers: _________________________________
Next: ____________________________________
```

---

## 🆘 Decision Tree

**Q: Where do I start?**  
A: Start with **P8** (config) + **P6** (class)

**Q: What if I don't know uncertainties?**  
A: Document where they should come from in P7, implement later

**Q: Can I do P4 without P7?**  
A: No - P4 needs uncertainties from P7

**Q: What if something's not clear?**  
A: Check the full tracking document, section by section

**Q: How do I know it's working?**  
A: UDRM state changes (alpha/beta values), tracked in `udrm_states` dict

---

## 📞 Context to Keep

**Paper Reference:** Sheeraja's IJCNN 2026 UDRM  
**Algorithm:** Algorithm 1 - Process transition with uncertainty  
**Domain:** Quantum path optimization (multi-arm bandit variant)  
**Framework:** Your QuantumExperimentRunner-based orchestrator  

---

## 🎬 Ready to Start?

**Next Session:**
1. Open full tracking document (UDRM-Integration-Tracking-Plan.md)
2. Start with Point P8 (Config Integration)
3. Update this quick ref with your progress
4. Move to P6 (UDRM Class) once P8 ✅

Good luck! 🚀
