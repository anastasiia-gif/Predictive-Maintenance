# Assignment 2 — Added Value of the Predictive-Maintenance System

**Course:** AIS — Applied Intelligent Systems
**Scope:** Model_00AFC, Model_06BQ, Model_37LB

---

## 1. Cost model

| Event | Cost |
|---|---|
| Planned motor replacement | **€4 500** |
| Unexpected failure (replacement **+** production downtime) | €4 500 + €3 500 = **€8 000** |

The predictive-maintenance (PdM) system does not change *how often* a motor is replaced — a motor is
replaced roughly every 15 000 parts either way. Its value comes from **converting unexpected failures
(€8 000) into planned replacements (€4 500)**, saving the **€3 500** downtime cost each time it catches a
failure in advance.

---

## 2. Inputs measured from the data

* **Baseline failure rate** `p` — the share of motor cycles that end in an *unexpected failure* under the
  current "replace at 15 000 parts" policy (measured over **all** files of each model).
* **Catch rate** `c` — the share of those failures the PdM system warns about **before** they happen
  (measured on **held-out** machines only, so no data leakage). The warning is issued when the predicted
  RUL drops to 250 parts (UR2).
* **Mean warning lead** — how many parts before the end the warning fires (used for the small
  "early-replacement" cost).

| Model | Motor cycles | Baseline failure rate (p) | Catch rate (c) | Mean warning lead |
|---|---:|---:|---:|---:|
| Model_00AFC | 396 | **36.6 %** | 100 % | 265 parts |
| Model_06BQ | 825 | **11.4 %** | 100 % | 243 parts |
| Model_37LB | 561 | **83.1 %** | 100 % | 274 parts |

The three machines differ enormously in how often the motor fails early — from 1 in 9 cycles
(Model_06BQ) to 5 in 6 cycles (Model_37LB) — which directly drives how much value PdM delivers.

---

## 3. Calculation

For one motor cycle:

```
Cost WITHOUT PdM   = 4500 + p · 3500
Cost WITH PdM      = 4500 + p·(1−c) · 3500  +  early_replacement_cost
Saving per cycle   = Cost_without − Cost_with
                   = p · c · 3500  −  early_replacement_cost
```

`p·(1−c)` is the **residual** failure rate: failures the system misses. The
`early_replacement_cost` accounts for replacing the motor ~275 parts early
(`lead / 15000 · €4 500 ≈ €80`), i.e. slightly more motors are bought over time.

### Worked example — Model_37LB
```
Cost without PdM = 4500 + 0.831 · 3500              = €7 407 / cycle
Residual failures = 0.831 · (1 − 1.00)   = 0.000   (all failures caught in time)
Early-replacement = 274/15000 · 4500               ≈ €82
Cost with PdM    = 4500 + 0.000 · 3500 + 82         = €4 582 / cycle
Saving           = 7 407 − 4 582                    = €2 825 / cycle
```

---

## 4. Results — value per machine type

| Model | Cost/cycle **without** PdM | Cost/cycle **with** PdM | **Saving / cycle** | Saving / 1 000 motors |
|---|---:|---:|---:|---:|
| Model_00AFC | €5 782 | €4 580 | **€1 202** | €1.20 M |
| Model_06BQ | €4 899 | €4 573 | **€326** | €0.33 M |
| Model_37LB | €7 407 | €4 582 | **€2 825** | €2.83 M |

*(Catch rate = 100 % on held-out machines with the tuned model, so residual unexpected failures ≈ 0; the
small remaining cost with PdM is just the ~275-part early replacement.)*

The saving is almost entirely explained by the baseline failure rate: the more often a machine's motor
fails unexpectedly today, the more downtime the system prevents.

---

## 5. Recommendations

1. **Deploy first on Model_37LB.** With 83 % of motors failing early today, PdM saves **~€2 825 per
   motor cycle** and eliminates the costly downtime — by far the highest return.
2. **Deploy on Model_00AFC** as well (~€1 202 / cycle). A one-third failure rate makes the business case
   clearly positive.
3. **Model_06BQ is lower-priority but still worthwhile** (~€326 / cycle). Because its motors rarely fail
   early, the downtime avoided is smaller; the system essentially provides insurance plus a modest saving.
4. **Keep the scheduled-interval fallback.** Always replacing at 15 000 parts if no earlier warning has
   fired guarantees no motor is ever run to destruction — a safety net even though the tuned model caught
   100 % of failures in time on the held-out data.
5. **Tune the warning lead if downtime is critical.** Raising the warning threshold above 250 parts
   catches even more failures at the cost of replacing motors slightly earlier — a cheap trade given
   downtime costs €3 500 versus ~€0.30 of motor life per part.

---

## 6. Assumptions & caveats

* Failure and catch rates are estimated from the provided data; real-world value depends on these staying
  stable. Model_06BQ in particular has few failures, so its catch rate (100 % on the test set) carries more
  statistical uncertainty.
* The €3 500 downtime cost is treated as fixed per failure; in practice it may vary with order backlog.
* "Early-replacement cost" assumes wasted motor life scales linearly with the fixed €4 500 motor price.
* Savings are **per motor cycle** (~15 000 parts). Multiply by the number of cycles per machine per year
  to obtain an annual figure.
* **No timestamps in the data.** The logs are indexed by a sequential part counter, not by date/time, so
  value is expressed *per motor cycle*; converting to an annual figure requires the real production rate
  (parts per day), which is not in the dataset.
* **Class imbalance.** Failure vs maintenance events are imbalanced and the balance differs per machine
  (Model_06BQ: only ~11 % failures, 94 events in total; Model_37LB: ~83 %). Estimates on the scarce side
  (failures for Model_06BQ, maintenance for Model_37LB) therefore carry more statistical uncertainty. The
  warning model is evaluated with a confusion matrix and precision/recall (Assignment 1, §6.2) rather than
  raw accuracy, because the near-end-of-life "warning zone" is only ~1.7 % of all rows.
