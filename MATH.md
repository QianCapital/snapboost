# The Mathematics of SnapBoost

This note develops SnapBoost from functional gradient boosting, derives its
Newton working-response problem, describes the learner-specific mathematics,
and compares the resulting algorithm with traditional gradient boosting and
XGBoost. Equations marked as implementation details describe this repository;
the general Heterogeneous Newton Boosting Machine (HNBM) formulation is broader.

## 1. Setup and notation

Given observations $(x_i,y_i,w_i)_{i=1}^n$, with $w_i\geq0$, consider the
empirical risk

$$
\mathcal R(F)=\sum_{i=1}^{n}w_i\ell(y_i,F(x_i)).
$$

The model is an additive expansion

$$
F_M(x)=F_0+\sum_{m=1}^{M}\eta_m f_m(x),
\qquad f_m\in\mathcal H_{k_m},
$$

where $\mathcal H_{k_m}$ is one member of a heterogeneous pool of hypothesis
classes. In this implementation the pool can contain:

- regression trees of several maximum depths;
- ridge regression over random Fourier features (RFFs), optionally at several
  kernel bandwidths and for RBF or Laplacian kernels; and
- an optional ridge model on the original standardized features.

$F(x)$ is a raw score. For regression it is the prediction itself. For binary
classification it is a log-odds score transformed by the sigmoid.

## 2. From gradient boosting to Newton boosting

### 2.1 Functional gradient boosting

Ordinary gradient boosting follows steepest descent in function space. At
round $m$, its pseudo-residuals are

$$
u_i=-\left.\frac{\partial\ell(y_i,F)}{\partial F}
\right|_{F=F_{m-1}(x_i)}=-g_i.
$$

A weak learner is fit to $(x_i,u_i)$ and then added to the ensemble. This uses
the slope of the loss but ignores its local curvature.

### 2.2 The second-order surrogate

Let $q_i=f(x_i)$ be a proposed correction. A second-order Taylor expansion is

$$
\ell(y_i,F_{m-1}(x_i)+q_i)
\approx \ell_i+g_iq_i+\frac12 h_iq_i^2,
$$

where

$$
g_i=\left.\partial_F\ell(y_i,F)\right|_{F_{m-1}(x_i)},
\qquad
h_i=\left.\partial_F^2\ell(y_i,F)\right|_{F_{m-1}(x_i)}.
$$

Completing the square gives

$$
g_iq_i+\frac12h_iq_i^2
=\frac12h_i\left(q_i+\frac{g_i}{h_i}\right)^2
-\frac{g_i^2}{2h_i}.
$$

The last term is independent of $q_i$. Consequently, minimizing the quadratic
surrogate over a learner class is equivalent to

$$
f_m\in\arg\min_{f\in\mathcal H_{k_m}}
\sum_{i=1}^{n}w_i h_i
\left(r_i-f(x_i)\right)^2,
\qquad r_i=-\frac{g_i}{h_i}.
\tag{1}
$$

The factor $1/2$ has no effect on the minimizer. Equation (1) is the central
computation in SnapBoost: the Hessian supplies both the Newton denominator and
the sample weights used to fit the base regressor. The code uses
`newton_target = -g / h` and `fit_weight = h * sample_weight`.

If the learner class could interpolate every $r_i$, this would be a pointwise
Newton step. A constrained weak learner instead projects that step onto a
structured function class.

### 2.3 Shrinkage and the actual update

After fitting $f_m$, SnapBoost updates

$$
F_m(x)=F_{m-1}(x)+\eta_m f_m(x).
\tag{2}
$$

By default $\eta_m=\eta$, the configured `learning_rate`. With
`line_search=True`, this repository evaluates the finite grid

$$
\eta_m\in\eta\{0.25,0.5,1,1.5,2\}
$$

and chooses the value giving the smallest current training loss. This is a
discrete search, not an exact continuous line search.

## 3. Losses, derivatives, and initial scores

### 3.1 Squared-error regression

The implementation defines

$$
\ell(y,F)=(F-y)^2,
\qquad g=2(F-y),
\qquad h=2.
$$

Therefore

$$
r=-\frac gh=y-F,
$$

so Newton boosting reduces to fitting the ordinary residual. The initial score
is the weighted mean

$$
F_0=\frac{\sum_iw_i y_i}{\sum_iw_i}.
$$

### 3.2 Binary logistic classification

Internally the two classes are mapped to $y\in\{-1,+1\}$. With

$$
\ell(y,F)=\log(1+e^{-yF}),
$$

write $a=\sigma(-yF)$, where $\sigma(t)=1/(1+e^{-t})$. Then

$$
g=-ya,
\qquad h=a(1-a)=\sigma(-yF)\sigma(yF),
$$

and the Newton response is

$$
r=\frac{y}{1-a}=\frac{y}{\sigma(yF)}.
$$

The Hessian becomes small for confidently classified observations, reducing
their weighted influence. If

$$
\hat p=\frac{\sum_iw_i\mathbf1[y_i=+1]}{\sum_iw_i},
$$

the initial score is the clipped empirical log-odds

$$
F_0=\log\frac{\hat p}{1-\hat p}.
$$

Predicted positive-class probability is

$$
P(y=+1\mid x)=\sigma(F_M(x)).
$$

### 3.3 Pseudo-Huber regression

For residual $e=F-y$ and transition scale $\delta>0$,

$$
\ell_\delta(y,F)=\delta^2
\left(\sqrt{1+(e/\delta)^2}-1\right),
$$

$$
g=\frac{e}{\sqrt{1+(e/\delta)^2}},
\qquad
h=\left(1+(e/\delta)^2\right)^{-3/2}.
$$

Large residuals have small curvature and hence small fitting weight, producing
a smooth robust alternative to squared error. The implementation floors $h$
at machine epsilon for numerical safety and initializes $F_0$ to the weighted
mean.

### 3.4 Quantile regression

For quantile level $\tau\in(0,1)$ and residual $e=y-F$, the pinball loss is

$$
\rho_\tau(e)=\max\{\tau e,(\tau-1)e\}.
$$

It is not twice differentiable. This implementation uses

$$
g=\begin{cases}1-\tau,&F\geq y,\\-\tau,&F<y,\end{cases}
\qquad h=1
$$

as a unit-Hessian working approximation. Thus this option uses the same
weighted fitting machinery but is not a literal Newton step for pinball loss.
$F_0$ is the weighted empirical $\tau$-quantile.

## 4. The heterogeneous SnapBoost step

Let tree depths be $d\in\{d_{\min},\ldots,d_{\max}\}$, let $p_T$ be
`p_tree`, and let $p_L$ be `p_linear`. The remaining mass

$$
p_K=1-p_T-p_L
$$

is assigned to RFF candidates. With $D_T=d_{\max}-d_{\min}+1$ tree
candidates and $D_K$ kernel-type/bandwidth pairs, the default random strategy
uses

$$
P(k_m=\text{tree depth }d)=\frac{p_T}{D_T},
$$

$$
P(k_m=\text{kernel candidate }j)=\frac{p_K}{D_K},
\qquad
P(k_m=\text{linear})=p_L.
\tag{3}
$$

The default parameters have $p_T=0.9$, $p_L=0$, depths 2 through 4, and one
RBF candidate. Thus each tree depth has probability $0.3$ and the RFF learner
has probability $0.1$ at every round.

Random selection is part of the HNBM design: it avoids fitting every expensive
hypothesis class every round and diversifies the ensemble. With
`selection_strategy="greedy"`, all positive-probability candidates are fit to
the same Newton problem and SnapBoost selects

$$
(k_m,\eta_m)\in\arg\min_{k,\eta_k}
\sum_iw_i\ell\left(y_i,F_{m-1}(x_i)+\eta_k f_{m,k}(x_i)\right).
\tag{4}
$$

The configured probabilities then determine candidate eligibility, but do not
otherwise weight the choice in (4).

When `subsample=s<1`, the learner fit uses a reproducible sample without
replacement of size $\lceil sn_+\rceil$, where $n_+$ counts observations with
positive effective Hessian weight. Candidate predictions and selection loss
are still evaluated over the full training set.

## 5. Learner-specific subproblems

### 5.1 Decision trees

A regression tree partitions feature space into leaves
$R_1,\ldots,R_J$ and predicts

$$
f(x)=\sum_{j=1}^{J}c_j\mathbf1[x\in R_j].
$$

For a fixed partition, equation (1) gives the unregularized weighted leaf value

$$
c_j=
\frac{\sum_{i:x_i\in R_j}w_i h_i r_i}
     {\sum_{i:x_i\in R_j}w_i h_i}
=-\frac{\sum_{i:x_i\in R_j}w_i g_i}
        {\sum_{i:x_i\in R_j}w_i h_i}.
\tag{5}
$$

SnapBoost delegates partition construction to scikit-learn's
`DecisionTreeRegressor`, using weighted squared-error reduction on the working
response. Depth and minimum-leaf-size constraints regularize tree structure;
optional feature sampling and monotonic constraints further restrict it.

### 5.2 RBF random Fourier features

The RBF kernel is

$$
k_{\mathrm{RBF}}(x,x')=\exp(-\gamma\lVert x-x'\rVert_2^2).
$$

Bochner's theorem represents a stationary positive-definite kernel as an
expectation over Fourier bases. Drawing

$$
\omega_j\sim\mathcal N(0,2\gamma I),
\qquad b_j\sim\operatorname{Unif}(0,2\pi),
$$

defines the $D$-dimensional map

$$
\phi_j(x)=\sqrt{\frac2D}\cos(\omega_j^\top x+b_j),
\qquad
\phi(x)^\top\phi(x')\approx k_{\mathrm{RBF}}(x,x').
\tag{6}
$$

By default, features are standardized before (6), which is important because
kernel distances depend on feature scale. A fresh reproducible random basis is
derived for every boosting round.

Let $\Phi_{ij}=\phi_j(x_i)$, $W=\operatorname{diag}(w_i h_i)$, and
$r=(r_1,\ldots,r_n)^\top$. The RFF ridge coefficients solve

$$
\hat\beta=\arg\min_\beta
(r-\Phi\beta)^\top W(r-\Phi\beta)+\alpha\lVert\beta\rVert_2^2,
$$

with normal-equation solution

$$
\hat\beta=(\Phi^\top W\Phi+\alpha I)^{-1}\Phi^\top Wr.
\tag{7}
$$

The fitted correction is $f(x)=\phi(x)^\top\hat\beta$ (with the intercept
handling supplied by scikit-learn's ridge estimator). In the code, positive
weights are normalized to have mean one before ridge fitting. This leaves an
unregularized weighted least-squares solution unchanged, while fixing the
effective scale of $\alpha$ across rounds.

### 5.3 Laplacian random features

The optional Laplacian kernel is

$$
k_{\mathrm{Lap}}(x,x')=\exp(-\gamma\lVert x-x'\rVert_1).
$$

Its spectral distribution factorizes into Cauchy variables, so the
implementation samples each coordinate of $\omega_j$ independently as

$$
\omega_{qj}\sim\operatorname{Cauchy}(0,\gamma)
$$

and uses the same cosine map as (6).

### 5.4 Raw-feature linear ridge

When $p_L>0$, the optional linear candidate has

$$
f(x)=\beta_0+x^\top\beta,
$$

and solves weighted ridge regression on standardized original features. It can
capture an inexpensive global linear correction without Monte Carlo features.

### 5.5 Exact kernel-ridge variant

The separately exposed exact-kernel estimators replace the RFF approximation
with an RBF Gram matrix $K$, where $K_{ij}=k(x_i,x_j)$. In the conventional
unweighted form, kernel ridge solves

$$
\hat a=(K+\alpha I)^{-1}r,
\qquad f(x)=\sum_i\hat a_i k(x_i,x).
$$

This avoids random-feature approximation error but generally requires
$O(n^2)$ memory and up to $O(n^3)$ factorization time. The default RFF path is
linear in $n$ for fixed feature dimension $D$, apart from the ridge solve.

## 6. Complete algorithm

For the recommended classifier and regressor classes, training can be written
as follows:

1. Map binary labels to $\{-1,+1\}$ if necessary and compute $F_0$.
2. For $m=1,\ldots,M$:
   1. Evaluate $g_i$ and $h_i$ at $F_{m-1}(x_i)$.
   2. Form $r_i=-g_i/h_i$ and $\tilde w_i=w_i h_i$.
   3. Sample one learner class using (3), or fit all eligible classes and use
      (4).
   4. Fit the chosen regressor to $(x_i,r_i)$ with weights $\tilde w_i$.
   5. Set $\eta_m=\eta$, or select it from the line-search grid.
   6. Update the raw prediction using (2).
3. If validation early stopping is enabled, retain the ensemble size having
   the best validation loss according to `min_delta` and patience.

The final raw prediction is

$$
F_M(x)=F_0+\sum_{m=1}^{M'}\eta_m f_m(x),
$$

where $M'\leq M$ if training stops early.

## 7. Comparison with traditional boosting

### 7.1 AdaBoost

AdaBoost constructs a weighted vote of classifiers and updates observation
weights exponentially according to classification mistakes. Its classical
exponential loss is

$$
\ell(y,F)=e^{-yF}.
$$

SnapBoost instead optimizes differentiable objectives through gradients and
Hessians, fits real-valued regression corrections, and can mix structurally
different learners. It is therefore closer to gradient/Newton boosting than to
the original reweight-and-vote interpretation of AdaBoost.

### 7.2 First-order gradient boosting

First-order boosting fits $-g_i$. SnapBoost fits $-g_i/h_i$ with weight
$w_i h_i$. The distinction disappears up to a constant for squared error,
because $h_i=2$, but matters for logistic and pseudo-Huber objectives where
curvature varies by observation.

### 7.3 Conventional Newton tree boosting

Newton tree boosting uses the same local quadratic idea but always projects the
Newton direction onto a tree class. SnapBoost generalizes the projection:
$\mathcal H_{k_m}$ can be a tree, a smooth kernel approximation, or a linear
model. This changes the inductive bias without changing the common derivative
interface.

### 7.4 KTBoost-style greedy heterogeneity

A greedy heterogeneous method can fit both a tree and a kernel model on every
round and keep the better correction. SnapBoost's default instead samples one
hypothesis subclass, trading a potentially less greedy individual step for
lower round cost and randomized diversity. This repository's optional greedy
strategy provides the other behavior when its extra fitting cost is acceptable.

## 8. Detailed comparison with XGBoost

XGBoost also begins with a second-order approximation. For a new tree $f_t$ it
writes, up to constants,

$$
\widetilde{\mathcal L}^{(t)}
=\sum_i\left[g_i f_t(x_i)+\frac12h_i f_t(x_i)^2\right]
+\Omega(f_t),
$$

with tree regularizer commonly expressed as

$$
\Omega(f)=\gamma_T T+\frac12\lambda\sum_{j=1}^{T}c_j^2,
$$

where $T$ is the number of leaves and $c_j$ is leaf $j$'s score. For a fixed
tree structure, define

$$
G_j=\sum_{i\in R_j}g_i,
\qquad H_j=\sum_{i\in R_j}h_i.
$$

The optimal regularized leaf score and corresponding structure score are

$$
c_j^*=-\frac{G_j}{H_j+\lambda},
$$

$$
\widetilde{\mathcal L}^{(t)}(q)
=-\frac12\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda}+\gamma_T T.
$$

For left and right child statistics $(G_L,H_L)$ and $(G_R,H_R)$, XGBoost's
canonical split gain is

$$
\operatorname{Gain}=\frac12\left[
\frac{G_L^2}{H_L+\lambda}
+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma_T.
\tag{8}
$$

SnapBoost and XGBoost are therefore closely related at the surrogate level but
differ in how they minimize it:

| Aspect | SnapBoost in this repository | XGBoost |
|---|---|---|
| Per-round approximation | Newton working response, Hessian-weighted least squares | Direct second-order objective in gradient/Hessian statistics |
| Learner class | Heterogeneous pool: trees, RFF ridge, optional linear ridge | Trees in standard `gbtree` mode; linear booster is a separate booster choice |
| Choice per round | Random class sampling by default; optional greedy comparison | Greedy/approximate tree construction and split selection |
| Tree leaf values | Produced by weighted regression-tree fitting, equivalent to (5) for a fixed partition | Regularized value $-G_j/(H_j+\lambda)$ |
| Structural regularization | Depth, minimum leaf size, feature sampling, optional monotonicity | Depth/leaves, minimum child weight, split penalty $\gamma_T$, L1/L2 leaf penalties, and more |
| Smooth nonlinear component | Explicit RFF or exact kernel-ridge learner | Not part of the standard tree booster |
| Step size | Fixed shrinkage or small discrete loss search | Usually global learning-rate shrinkage after tree optimization |
| Computational emphasis | Avoid fitting all families under random selection; RFF scales with fixed $D$ | Highly optimized histogram/approximate tree construction |

Observation weights can be folded into both methods by replacing
$g_i,h_i$ with $w_i g_i,w_i h_i$ in aggregated statistics. SnapBoost expresses
this as the effective fitting weight $w_i h_i$ while leaving the Newton target
$-g_i/h_i$ unchanged.

The practical inductive-bias difference is central. A tree produces piecewise
constant, axis-aligned corrections and is strong at thresholds and feature
interactions. An RFF ridge model produces a smooth, distributed correction and
is strong when nearby standardized points should have similar predictions.
SnapBoost can add both kinds of functions to one ensemble; standard XGBoost's
tree booster repeatedly adds only the first kind.

## 9. Regularization and approximation effects

SnapBoost has several interacting forms of regularization:

- **Shrinkage:** smaller $\eta_m$ reduces every learner's immediate effect and
  usually requires more rounds.
- **Tree capacity:** depth, leaf-size, feature-sampling, and monotonicity limits
  restrict the tree projection of the Newton step.
- **Ridge penalty:** $\alpha\lVert\beta\rVert_2^2$ controls the size of smooth
  or linear corrections.
- **Kernel scale:** $\gamma$ controls locality. Larger RBF $\gamma$ makes the
  kernel decay faster with distance.
- **RFF dimension:** larger $D$ generally reduces Monte Carlo kernel
  approximation error at greater memory and compute cost.
- **Row subsampling and learner sampling:** these inject randomness and reduce
  individual-round cost.
- **Early stopping:** validation loss selects the effective ensemble length.

RFF approximation is unbiased in the sense that, for the sampling scheme in
(6),

$$
\mathbb E[\phi(x)^\top\phi(x')]=k(x,x'),
$$

while its finite-$D$ variance decreases as the number of random features grows.
Boosting draws fresh bases across rounds, so the final model combines multiple
finite-dimensional kernel approximations rather than relying on one fixed map.

## 10. What is specific to this implementation

The mathematical core is equation (1), but several details are worth keeping
separate from a generic description of the SnapBoost paper:

- The default non-tree learner is scikit-learn ridge over random Fourier
  features, not an exact kernel solve.
- The default strategy samples one learner; greedy selection and discrete line
  search are opt-in extensions.
- The current default pool uses tree depths 2, 3, and 4 with total probability
  0.9, plus one RBF-RFF learner with probability 0.1.
- Classification is binary and uses logistic loss.
- Regression supports squared error plus optional pseudo-Huber and quantile
  objectives; quantile uses a unit-Hessian working approximation.
- The recommended RFF path standardizes inputs and normalizes positive fitting
  weights before ridge fitting.
- The exact-kernel classes are optional alternatives with materially greater
  time and memory requirements.

For the literature underlying HNBM, SnapBoost, random Fourier features, and
XGBoost, see [REFERENCES.md](REFERENCES.md) and [CITATION.bib](CITATION.bib).
