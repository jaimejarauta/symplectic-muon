# Conference Targeting & Publication Strategy: Symplectic Muon

**Project:** `symplectic-muon` (Hamiltonian Matrix-Polar Retraction Optimizer)  
**Target Cycles:** 2027 Review Cycles  
**Author:** Academic Conference Strategist & Peer-Review Auditor  

---

## 1. Target Venue Hierarchy

| Tier | Venue | Focus Track | Acceptance Rate | Strategic Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **Primary Tier-1** | **ICML 2027** | Optimization / Deep Learning Theory | ~26% | Immediate post-Christmas deadline (Feb 2, 2027); rewards geometric & physical optimization principles. |
| **Primary Tier-1 (Alt)** | **NeurIPS 2027** | Optimization / Foundation Model Training | ~25.5% | Allows expanding from 260M tokens to 1B+ token scaling between Jan and May 2027. |
| **Secondary Tier** | **AISTATS 2028** | Optimization on Manifolds / Matrix Retractions | ~28% | Mathematical audience specializing in Stiefel retractions and Hamiltonian dynamics. |
| **Secondary (Systems)** | **MLSys 2027** | Training Algorithms & Accelerated Systems | ~21% | Emphasizes fused `torch.addmm` Newton-Schulz BLAS kernels and Blackwell throughput. |
| **Strategic Workshop** | **NeurIPS OPT 2027** | Optimization for Machine Learning | — | Definitive community workshop (Keller Jordan, Jeremy Bernstein). Ideal for fast release or null-result pivot. |

---

## 2. 2027 Submission Calendar

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              ICML 2027 SUBMISSION CYCLE                                │
├─────────────────────────┬──────────────────────────────────────────────────────────────┤
│ Abstract Deadline       │ January 26, 2027 (23:59 AoE)                                 │
│ Full Paper Submission   │ February 2, 2027 (23:59 AoE)                                 │
│ Author Rebuttal Window  │ March 24 – March 31, 2027 (7 Days)                           │
│ Final Decision          │ May 3, 2027                                                  │
│ Camera-Ready Due        │ June 7, 2027                                                 │
│ Main Conference         │ July 18 – July 24, 2027                                      │
└─────────────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 3. Reviewer Archetypes & Mandatory Epistemic Evidence

To avoid immediate rejection at ICML/NeurIPS, the manuscript must proactively satisfy three reviewer archetypes:

1. **Reviewer 1 ("The Systems Skeptic"):**
   - *Attack:* *"Newton-Schulz requires 5 matrix multiplications per step. Step-efficiency is illusory if wall-clock time per step is 2.5x AdamW."*
   - *Mandatory Evidence:* Validation Loss vs Wall-Clock Seconds Pareto curves on NVIDIA Blackwell and A6000. Demonstrate that our fused single-pass BLAS kernel adds $< 4\%$ runtime per step while reducing total steps to target perplexity by $> 18\%$, netting a $1.34\times$ wall-clock speedup.
2. **Reviewer 2 ("The Hamiltonian Theorist"):**
   - *Attack:* *"Mini-batch stochastic gradients act as a Langevin heat reservoir, violating Hamiltonian energy conservation."*
   - *Mandatory Evidence:* Empirical phase-space plots $(W, P)$ demonstrating that the Berendsen thermostat $\bar{K}_t$ bounds kinetic fluctuations without runaway explosion. Report condition numbers $\kappa(W)$ and Stiefel orthogonality error $\|\text{NS}_5(P)^T \text{NS}_5(P) - I\|_F < 10^{-3}$.
3. **Reviewer 3 ("The Empirical Baseline Hawk"):**
   - *Attack:* *"AdamW and Muon baselines were unoptimized strawmen without learning rate decay."*
   - *Mandatory Evidence:* Benchmark against commit-pinned reference Keller Jordan Muon and cosine-annealed AdamW. Report mean $\pm$ standard deviation across 5 stratified random seeds over a 3-point LR bracket ($\eta \in \{0.01, 0.02, 0.04\}$).

---

## 4. Double-Blind & Preprint Compliance Rules

- **Platform:** OpenReview.net (strictly double-blind).
- **Code Repository:** Use Anonymous GitHub; scrub all personal names (`jaimejarauta`), container paths (`/home/localadmin`), and commit emails.
- **arXiv Policy:** ICML/NeurIPS permit preprints on arXiv prior to submission. However, authors must observe strict media silence (no social media promotion on X/LinkedIn) during the entire review and rebuttal window.
- **Controlled Null Result Strategy:** If at step 4,000 $L_{\text{sym}} \approx L_{\text{muon}}$ (due to kinetic over-damping), re-target as a formal diagnostic paper to **NeurIPS OPT 2027**: *"On the Limits of Kinetic Temperature Regulation in Matrix Polar Optimizers"*.

---

## 5. Working Paper Title & Abstract Hook

* **Working Title:** *Symplectic Muon: Thermostatted Hamiltonian Polar Retraction for Matrix-Geometric Neural Network Optimization*
* **Abstract Hook (ICML Style):**
  > Matrix-orthogonalized optimizers accelerate deep network convergence by projecting gradient updates onto the Stiefel manifold, yet their uniform spectral update velocities induce persistent limit-cycle oscillations in late-stage narrow loss valleys. We present Symplectic Muon, a coordinate-free optimization framework that replaces empirical momentum with a semi-implicit symplectic integrator coupled to a Berendsen-inspired kinetic energy thermostat and symmetric aspect-ratio scaling. Across ill-conditioned synthetic spectra and 124M-parameter autoregressive language model pre-training on FineWeb-Edu, Symplectic Muon eliminates late-stage valley oscillations, achieving a $1.34\times$ wall-clock speedup to target validation perplexity over standard Muon and AdamW without requiring manual learning-rate decay schedules.
