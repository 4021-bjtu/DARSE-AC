# Paper-to-Code Reconstruction / 论文到代码对照

This document records how the DARSE-AC algorithm maps to the code in this repository: the discrete conventions, the HQS/FFT derivation, every implementation choice where the paper leaves details open, and the corresponding reading notes. 中文：本文档记录 DARSE-AC 论文算法到本仓库代码的映射：离散约定、HQS/FFT 推导、论文未明确处的实现选择，以及对应研读笔记。

## Source / 正式文

Shuo Tan, Lingye Dong, Limei An. *The Regularization and Deconvolution Algorithm Combining Salient Edges and Average Curvature in High-quality Visual Communication*. Information Technology and Control 54(1), 234–253 (2025). DOI: [10.5755/j01.itc.54.1.38163](https://doi.org/10.5755/j01.itc.54.1.38163).

Formal PDF: https://itc.ktu.lt/index.php/ITC/article/view/38163/16886

The PDF was downloaded and parsed. Body extraction starts at PDF page 2; no first-page screenshot is used. `source/provenance.json` records its SHA256. Printed page = PDF page + 233. Extracted equations are not reliable on their own; pages 239–240 were checked visually. 中文：公式须以正式版页面为准，文本提取有双栏与符号错位。

## Mathematical Contract / 数学约定

Use grayscale float64 images in [0,1], unit pixel spacing, spatial sums (not means), periodic forward differences `D`, exact adjoint `D*`, centered odd-size kernels, unnormalised NumPy FFT / normalised inverse FFT.

From Eq.(5), image block:

```text
||k*I-A||² + lambda ||D I-D z||² + xi ||D I||_0 + theta Q(I).
kernel block: ||k*D z-D A||² + tau ||k||².
```

From Eq.(6), split `alpha ~ I`, `beta ~ D I` and add `theta1 ||alpha-I||² + xi1 ||beta-D I||²`.

```text
beta(p) = D I(p) if ||D I(p)||² > xi/xi1, else 0.
I_hat = (conj(K) A_hat + lambda |D|² z_hat
         + xi1 FFT(D* beta) + theta1 alpha_hat)
        / (|K|² + (lambda+xi1)|D|² + theta1).
K_hat = (conj(Zx) Ax + conj(Zy) Ay) / (|Zx|²+|Zy|²+tau).
```

联合梯度L0按像素计数，而非分别计数横纵分量；阈值等号处取零。FFT图像更新不裁剪，保证固定辅助变量时确为二次问题的解；最终输出裁剪至[0,1]。核解裁回指定支持域，再负值置零、和归一化，退化为全零时回退中心脉冲。真值核绝不传入 `deblur`。

## Substitutions / 逐项替代

| Paper evidence | Implemented choice / 本次实现 | Consequence / 影响 |
|---|---|---|
| Eq.(3)–(4), p.238: mean-curvature regularisation | Eq.(8) stencil `[-1,5,-1;5,-16,5;-1,5,-1]/16` with surrogate `Q(I)=sum(abs(C*I))` | Linear AC surrogate, **not exact nonlinear graph mean curvature** / 不是精确非线性平均曲率 |
| p.239–240: four half-window projection filter | Dual projected-gradient prox of `.5||alpha-I||² + theta/(2 theta1)||C alpha||_1`, 30 steps; accept only if surrogate energy does not increase | Does not reconstruct unspecified four windows / 没有冒充未公开的四半窗口实现 |
| Eq.(9): condition conflicts with stated nonincrease and has unusual subscript | Direct numerical check of the explicitly defined surrogate energy | Does not assert the printed inequality is a proven descent test / 不声称补齐原证明 |
| Eq.(10) prints L0 inside an alleged quadratic; Eq.(11) Y_beta prints alpha components | Derive linear image solve from Eq.(6), using quadratic `xi1` and actual beta | Transparent interpretation, not literal transcription / 按一致目标推导并披露修正 |
| Eq.(12): self-guided / mutually guided filtering, mu=.01 | Three frozen-weight iterations, `w=1/max(abs(Dz),mu)`, solve `(Id+gamma D*wD)z=I` by SciPy CG, rtol=1e-6 | Auxiliary denominator update not fully specified; implemented as scalar-channel IRLS surrogate / 单通道自引导替代，不是已核实原滤波器 |
| Eq.(5): gradient-domain ridge kernel | Frequency ridge followed by support crop and nonnegative normalisation | Projection is not exact constrained least squares; no unreported centering or connected-component pruning / 不含隐式核重定位 |
| p.241: pyramid; flow t=5; p.242: four levels and ten alternations | Default 3 levels at 1/4,1/2,1, omit scales below 24 px; 5 outer iterations; 6 joint HQS steps | Compact CPU demo, explicitly differs from paper settings / 用于小图运行，非论文原日程 |
| p.241: nested penalty doubling | Joint penalty starts .02, doubles six times; resets each outer iteration; theta1=0 when curvature disabled | Different finite schedule, no convergence theorem / 不能宣称等同原嵌套极限 |
| p.247 parameters 4e-4,4e-3,4e-3,2,8e-4 | Same nominal edge,L0,curvature,ridge,guide defaults | Gamma-to-guide mapping is an interpretation; discretisation differs, numbers not necessarily transferable / 数值同名不等于行为一致 |
| p.248,250: shared final nonblind methods, incompletely specified | Gradient-Tikhonov nonblind solve with regularisation .002 using estimated kernel | Not paper patch prior or nonlinear highlight treatment / 最终非盲恢复替代 |
| Color input, grayscale kernel estimation then color restoration | CLI converts input to 8-bit grayscale and reports grayscale results only | Color pipeline, HDR and alpha handling are out of scope / 非彩色忠实复现 |
| Unknown boundary and preprocessing | Periodic convolution for synthesis and solver; no alignment; full and radius-cropped scores | Matched synthetic boundary is optimistic; real photos may have seam artefacts / 真实边界不能直接视作已验证 |
| Original experiments | Generated text+geometry, packaged skimage camera/coins; 2 kernels; Gaussian noise .005; seeds 2025,2026 | New local evaluation, not ICDAR/iNaturalist/GLADNet or cultural artifacts / 本次实测不等于文物实测 |
| ER and success not sufficiently defined | `ER_proxy=MSE(output,truth)/MSE(oracle_true_kernel,truth)`; success proxy <=2 | Never equate to 1.61 or 99.54%; oracle is a solver-specific reference, not a guaranteed bound / 不能与论文ER直接比较 |

## Trace Interpretation / 曲线解释

`energy_per_pixel` logs an Eq.(5)-inspired **image** energy after the kernel step, with current z; it excludes the separate kernel objective. Scales, z, and finite HQS penalties change. The saved trace is a diagnostic, not a single globally monotone objective. Tiny numerical gradients are counted nonzero above squared magnitude `1e-12`. A flat trace does not prove convergence, global optimality or text authenticity.

`no_edge_term` sets lambda=0 but still uses z for kernel estimation. `no_guidance` sets gamma=0 (z=I), a different intervention. `no_curvature` disables both curvature prox and alpha coupling. `no_l0` removes the L0 threshold cost but retains finite splitting. `single_scale` changes only pyramid levels. These are **newly defined ablations**, not a guessed A–E mapping of Fig.7.

## Manuscript Issues / 稿件问题

- Section 3 is non-neural optimisation; Section 4 mixes MATLAB with PyTorch, activation slope, learning rate, batches and training. No reproducible neural architecture is provided. 不应据此声称训练了深度网络。
- Actual sample IDs, paired blur construction, splits, ER denominator, success threshold and variance are incomplete. 数据集总规模不是实用样本数。
- Eq.(10)/(11), Eq.(9), flowchart references to (13)/(14), and repeated Table 4 numbering require clarification. 实验表按页面定位，不掩盖正式版排版问题。
- Table 1 timing/FFT prose and complexity statements conflict. Table 3 has an unexplained structural-similarity quantity above 1 and mismatched prose columns. Do not silently turn these into benchmark truths.
- ER is lower-is-better. A rising threshold-success curve likely reflects relaxed acceptance, not higher error being desirable; exact original protocol remains unclear.
- Figures 6 and 9 show finite empirical trajectories, not global convergence proofs. Strong noise and blind-kernel ambiguities remain limitations.

## Attribution / 归属

`assets/paper_fig8a_text_ablation.png` is a cropped excerpt of Fig.8(a), p.246, from the published paper by Shuo Tan, Lingye Dong and Limei An. It is not an output of this reconstruction. The publisher page refers to Lithuanian copyright law Articles 4–37; no CC licence was confirmed. Further distribution requires rights review. No licence granted for our code may be applied to the paper excerpt. This workspace has not been uploaded or published.

`results/**/comparison.png` and `success_proxy.png` are produced by the local scripts. The synthetic text/geometry is generated here and is explicitly not a cultural artifact. Camera/coins originate from scikit-image's packaged sample data; their original rights and provenance remain with their respective sources (see scikit-image documentation). This delivery does not assert a blanket open-source licence over all materials.
