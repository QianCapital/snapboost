# References

This document collects the primary literature and related resources for **SnapBoost** and the **Heterogeneous Newton Boosting Machine (HNBM)** framework implemented in this repository.

---

## Primary Paper

**SnapBoost: A Heterogeneous Boosting Machine**  
Thomas Parnell\*, Andreea Anghel\*, Małgorzata Łazuka, Nikolas Ioannou, Sebastian Kurella, Peshal Agarwal, Nikolaos Papandreou, Haralampos Pozidis  
\*Equal contribution.

- **Venue:** NeurIPS 2020 (34th Conference on Neural Information Processing Systems)
- **Pages:** 20872–20883 (NeurIPS 2020, Volume 33)
- **arXiv:** [2006.09745](https://arxiv.org/abs/2006.09745)
- **DOI:** [10.48550/arXiv.2006.09745](https://doi.org/10.48550/arXiv.2006.09745)
- **NeurIPS proceedings:** [Abstract & PDF](https://proceedings.neurips.cc/paper/2020/hash/7fd3b80fb1884e2927df46a7139bb8bf-Abstract.html)
- **IBM Research:** [Publication page](https://research.ibm.com/publications/snapboost-a-heterogeneous-boosting-machine)

### Summary

The paper introduces the **Heterogeneous Newton Boosting Machine (HNBM)**, a second-order boosting framework in which the base hypothesis class is sampled stochastically at each iteration from a fixed pool of subclasses. The authors derive global linear convergence rates under strong convexity and Lipschitz-gradient assumptions.

**SnapBoost** is a concrete HNBM realization that, at each boosting round, randomly selects either:

1. A **binary decision tree** with maximum depth drawn from a configurable range, or  
2. A **linear regressor with random Fourier features** (approximating kernel ridge regression).

The original implementation is in C++ (OpenMP, Eigen) with a scikit-learn-compatible Python API, distributed as part of [IBM Snap ML](https://snapml.readthedocs.io/en/latest/boosting_machines.html).

### Relation to This Repository

This Python package implements the HNBM / SnapBoost *idea* using scikit-learn base learners:

| Aspect | Original SnapBoost (NeurIPS 2020) | This repository |
|--------|-----------------------------------|-----------------|
| Tree learners | Histogram-based BDTs, depth sampled per round | `DecisionTreeRegressor` with depths in `[min_max_depth, max_max_depth]` |
| Non-tree learner | Linear regressor + random Fourier features | `RandomFourierRidgeRegressor` using `RBFSampler` + `Ridge` |
| Optimization | Custom C++ Newton boosting | Weighted least-squares fit to Newton direction (`-g/h`, weights `h`) |
| API | IBM Snap ML `BoostingMachine*` | scikit-learn-style `SnapBoost` / `HNBM` |

When citing work based on this package, cite the NeurIPS 2020 paper above. BibTeX entries are in [`CITATION.bib`](CITATION.bib).

---

## Official Implementations

| Resource | Description | Link |
|----------|-------------|------|
| IBM Snap ML | Production C++/GPU implementation with `BoostingMachineClassifier` and `BoostingMachineRegressor` | [Documentation](https://snapml.readthedocs.io/en/latest/boosting_machines.html) |
| IBM Snap ML (legacy docs) | Earlier SnapBoost API reference | [v1.6 manual](https://ibmsoe.github.io/snap-ml-doc/v1.6.0/manual.html) |

Key Snap ML parameters that mirror this repo: `tree_select_probability` (≈ `p_tree`), `min_max_depth`, `max_max_depth`, `gamma`, `regularizer` / `alpha`.

---

## Foundational Boosting

| Reference | Why it matters |
|-----------|----------------|
| Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine* | Functional gradient descent formulation used by XGBoost, LightGBM, and HNBM |
| Friedman (2002), *Stochastic Gradient Boosting* | Subsampling and stochasticity in boosting |
| Freund & Schapire (1995), *AdaBoost* | Original boosting algorithm |
| Schapire (1990), *The Strength of Weak Learnability* | Theoretical foundation for boosting |

---

## Newton / Second-Order Boosting

| Reference | Why it matters |
|-----------|----------------|
| Sigrist (2018), *Gradient and Newton Boosting for Classification and Regression* | Compares gradient vs. Newton boosting; motivates second-order updates |
| Karimireddy et al. (2018), *Global Linear Convergence of Newton's Method* | Convergence analysis referenced in the SnapBoost paper |

---

## Heterogeneous & Randomized Boosting

These works directly motivated the HNBM design:

| Reference | Link | Notes |
|-----------|------|-------|
| **Sigrist (2019), KTBoost** | [arXiv:1902.03999](https://arxiv.org/abs/1902.03999) | Learns both a tree and a kernel ridge regressor each round; picks the better fit. Major inspiration for heterogeneous ensembles. |
| **Cortes et al. (2014), DeepBoost** | [ICML 2014](https://proceedings.mlr.press/v32/cortes14.html) | Margin-based generalization bounds for heterogeneous ensembles; selects hypothesis subclasses to minimize the bound. |
| **Cortes et al. (2019), Regularized Gradient Boosting** | NeurIPS 2019 | Non-uniform sampling of hypothesis space without second-order information. |
| **Lu & Mazumder (2018), Randomized Gradient Boosting Machine** | [arXiv:1810.10158](https://arxiv.org/abs/1810.10158) | Uniform random selection of base hypotheses; extended by HNBM to non-uniform sampling with Newton steps. |

---

## Random Fourier Features & Kernels

| Reference | Link | Notes |
|-----------|------|-------|
| Rahimi & Recht (2007), *Random Features for Large-Scale Kernel Machines* | [NIPS 2007](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines) | Random Fourier feature approximation used in the original SnapBoost linear subclass |
| Current implementation | — | `RandomFourierRidgeRegressor` uses `RBFSampler` followed by weighted `Ridge`, matching the random-feature structure while remaining pure Python/scikit-learn |
| Optional exact-kernel variant | — | `SnapBoostKernelRidgeClassifier` and `SnapBoostKernelRidgeRegressor` use exact RBF `KernelRidge`; this costs substantially more memory and time on large datasets |

---

## Comparison Baselines (from SnapBoost Experiments)

| Framework | Reference | Link |
|-----------|-----------|------|
| XGBoost | Chen & Guestrin (2016) | [KDD 2016](https://dl.acm.org/doi/10.1145/2939672.2939785) |
| LightGBM | Ke et al. (2017) | [NeurIPS 2017](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) |
| CatBoost | Prokhorenkova et al. (2018) | [NeurIPS 2018](https://papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features) |
| KTBoost | Sigrist (2019) | [arXiv:1902.03999](https://arxiv.org/abs/1902.03999) |

---

## Full Bibliography (NeurIPS 2020 Paper)

The complete numbered reference list from the SnapBoost paper (46 entries) is reproduced below for convenience.

1. Ailerons data set — https://www.dcc.fc.up.pt/~ltorgo/Regression/ailerons.html  
2. Bank datasets — https://www.dcc.fc.up.pt/~ltorgo/Regression/bank.html  
3. Condition Based Maintenance of Naval Propulsion Plants — UCI ML Repository  
4. Credit card fraud detection — https://www.kaggle.com/mlg-ulb/creditcardfraud  
5. Eigen C++ linear algebra library — http://eigen.tuxfamily.org/  
6. Elevators data set — https://www.dcc.fc.up.pt/~ltorgo/Regression/elevators.html  
7. Kaggle — https://www.kaggle.com/  
8. Mercari price suggestion challenge — Kaggle  
9. Parkinsons data set — UCI ML Repository  
10. Pumadyn datasets — https://www.dcc.fc.up.pt/~ltorgo/Regression/puma.html  
11. Rossmann store sales — Kaggle  
12. Arik & Pfister (2019), TabNet — arXiv:1908.07442  
13. Benavoli et al. (2016), post-hoc tests — JMLR 17(5)  
14. Chen & Guestrin (2016), XGBoost — KDD  
15. Coraddu et al. (2014), naval propulsion CBM — J. Engineering for the Maritime Environment  
16. Cormen et al. (2009), *Introduction to Algorithms*, 3rd ed.  
17. Cortes, Mohri & Storcheus (2019), Regularized gradient boosting — NeurIPS  
18. Cortes, Mohri & Syed (2014), Deep boosting — ICML  
19. Demšar (2006), statistical comparisons of classifiers — JMLR 7  
20. Fish, Kun & Lelkes (2015), Fair boosting — FAT/ML workshop  
21. Freund (1995), boosting by majority — Information and Computation  
22. Freund & Schapire (1995), AdaBoost — ECML  
23. Friedman (2001), gradient boosting machine — Annals of Statistics  
24. Friedman (2002), stochastic gradient boosting — Computational Statistics & Data Analysis  
25. Grari et al. (2019), fair adversarial gradient tree boosting — arXiv:1911.05369  
26. Guillame-Bert & Teytaud (2018), exact distributed random forest — arXiv:1804.06755  
27. Ke et al. (2017), LightGBM — NeurIPS  
28. Ibragimov & Gusev (2019), minimal variance sampling — NeurIPS  
29. Iman & Davenport (1980), Friedman statistic correction — Communications in Statistics  
30. Jamieson & Talwalkar (2016), successive halving — AISTATS  
31. Karimireddy, Stich & Jaggi (2018), Newton convergence — arXiv:1806.00413  
32. Ke et al. (2019), DeepGBM — KDD  
33. Li (2008), two-step rejection for multiple hypotheses — J. Statistical Planning and Inference  
34. Li et al. (2019), privacy-preserving GBDT — arXiv:1911.04209  
35. Little et al. (2007), voice disorder detection — Biomedical Engineering Online  
36. Lu & Mazumder (2018), randomized gradient boosting — arXiv:1810.10158  
37. Mehta, Agrawal & Rissanen (1996), SLIQ — EDBT  
38. Popov, Morozov & Babenko (2019), NODE — arXiv:1909.06312  
39. Prokhorenkova et al. (2018), CatBoost — NeurIPS  
40. Rahimi & Recht (2007), random features — NeurIPS  
41. Schapire (1990), strength of weak learnability — Machine Learning  
42. Shafer, Agrawal & Mehta (1996), SPRINT — VLDB  
43. Sigrist (2018), gradient and Newton boosting — arXiv:1808.03064  
44. Sigrist (2019), KTBoost — arXiv:1902.03999  
45. Vanschoren et al. (2013), OpenML — SIGKDD Explorations  
46. Zhang, Si & Hsieh (2017), GPU-accelerated tree boosting — arXiv:1706.08359  

---

## How to Cite

```bibtex
@inproceedings{parnell2020snapboost,
  title     = {{SnapBoost}: A Heterogeneous Boosting Machine},
  author    = {Parnell, Thomas and Anghel, Andreea and {\L}azuka, Ma{\l}gorzata and Ioannou, Nikolas and Kurella, Sebastian and Agarwal, Peshal and Papandreou, Nikolaos and Pozidis, Haralampos},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {20872--20883},
  year      = {2020},
  url       = {https://proceedings.neurips.cc/paper/2020/hash/7fd3b80fb1884e2927df46a7139bb8bf-Abstract.html},
  eprint    = {2006.09745},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi       = {10.48550/arXiv.2006.09745}
}
```

Additional BibTeX entries for related work are in [`CITATION.bib`](CITATION.bib).
