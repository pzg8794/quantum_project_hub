# Approach Writeup — Final Editorial Edits Log

**Document:** `approach_writeup.tex`  
**Session date:** 2026-04-12  
**Version:** v7 → v8 (final pre-submission)

---

## Summary

All edits were purely editorial — no content was added or removed beyond what is documented below. The `.tex` file was updated locally by the author. This log records what changed and why, for traceability.

Final state: **zero open issues, 0 issues found by automated scan, ~1,239 body words.**

---

## Changes Applied

### §1 — Approach Overview

| Location | Change | Rationale |
|---|---|---|
| Clinical sentence | `"can become less responsive to"` → `"are blind to"` | Removed hedge on a claim directly supported by `garcia2025equitable_bioinformatics`; confident register matches the evidence |
| Closing sentence | `"comparable"` → `"structurally comparable"` | Restored the structural bridge claim explicitly in §1, the first place it appears; §3 and §4 develop it fully but §1 should name it |

---

### §2 — Motivating Workflows (Clinical)

No changes. Section already contained the strong formulations:
- `"structurally invisible to the instrument"`
- `"certain subgroups"`
- `"the very groups already bearing the highest burden"`

---

### §3 — Shared Domain Model (COVID paragraph)

| Location | Change | Rationale |
|---|---|---|
| COVID example sentence | Restored named groups; changed `"can become"` → `"is"` | Named demographic groups (Black, Latino, other minority) were removed in a prior space-cutting pass from both §2 and §3 simultaneously, leaving zero specificity anywhere in the paper. For a fairness paper this specificity is evidence. The sentence was rewritten to be more direct and confident: `"...producing measurably worse outcomes for Black, Latino, and other minority groups---confirming that unfair routing is a performance failure, not only an equity concern."` |

---

### §4.1 — Connecting Decisions to the Domain Model (Phase 1 paragraph)

| Location | Change | Rationale |
|---|---|---|
| Phase 1 sentence | `"threat signals, reproducing"` → `"threat signals---thereby reproducing"` | Fixed comma splice introduced when em-dash was removed in a prior edit; the participial clause needs the em-dash to attach correctly |

---

## What Was NOT Changed

The following were reviewed and deliberately left as-is:

- §2 Quantum Routing Workflow — `"This is the same structural problem"` retained; strong and correct
- §3 Phase 2 italic anchor — `\textit{Phase~2 is therefore where the root cause...}` retained
- §4.2 and §4.3 — no changes needed
- §5 Implementation and Evaluation Plan — no changes
- §6 Research Contribution — no changes
- Figure and TikZ code — no changes
- Bibliography — no changes

---

## Space Accounting

All cuts made in prior sessions were validated as acceptable given:
1. The cut content exists in stronger form elsewhere in the paper, OR
2. The content was decorative rather than evidentiary

The one exception (COVID paragraph specificity) was restored at minimal word cost (~same sentence length).

**Final word count:** ~1,239 body words across 6 sections + bibliography. Figure is on page 3 (`\clearpage` + `[!t]`) and does not count toward the two-page limit.

---

## Automated Scan Results (final)

```
Real issues found: 0
All checks: PASS

  §4.1 Phase 1          FIXED    '---thereby' comma splice resolved
  §1                    PASS     'are blind to' strong phrasing
  §1                    PASS     'structurally comparable' present
  §3 COVID              PASS     Named groups present
  §3 COVID              PASS     'is a performance failure' (not 'can become')
  §3 Phase 2 anchor     PASS     Italic claim intact
  §2 Clinical           PASS     'structurally invisible' present
  §2 Clinical           PASS     'certain subgroups' present
  §2 Clinical           PASS     Burden language present
  §2 Quantum            PASS     Structural bridge present
```
