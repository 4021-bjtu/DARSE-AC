# DARSE-AC

**Combining Salient Edges and Average Curvature Regularization for Blind Image Deconvolution**

Tan, S., Dong, L., An, L. (2025). *The Regularization and Deconvolution Algorithm Combining Salient Edges and Average Curvature in High-quality Visual Communication*. Information Technology and Control, 54(1), 234–253. [DOI: 10.5755/j01.itc.54.1.38163](https://doi.org/10.5755/j01.itc.54.1.38163)

---

## Research Problem

Images acquired in real applications — heritage documents, rubbings, text archives, low-light scenes — are often degraded by motion blur, defocus, and noise. Blind deconvolution aims to recover a sharp image from a single blurred observation, but the problem is severely ill-posed: both the latent image and an unknown blur kernel must be estimated simultaneously. Standard priors either over-smooth edges or amplify noise, especially for text-like structures with fine strokes and ornamentation.

## Method

DARSE-AC formulates image deblurring as a joint variational problem with two complementary regularizers:

```text
min_{I,k}  ‖k·I − A‖² + λ‖∇I − ∇z‖² + ξ‖∇I‖₀ + θ·Q(I)
min_k      ‖∇z·k − ∇A‖² + τ‖k‖²
```

- **Salient-edge prior** explicitly selects major structures `z`, preserving lettering/stroke boundaries while suppressing texture that harms kernel estimation;
- **Average-curvature (AC) regularization** views intensity as a height field and constrains bending energy in non-edge regions, reducing staircase artifacts and spurious ripples;
- A **HQS splitting scheme** introduces auxiliary variables, decomposing the objective into tractable subproblems;
- **FFT-based closed-form updates** solve the image subproblem in the frequency domain;
- The kernel is estimated in the gradient domain with ridge regularization, followed by nonnegativity and normalization;
- A **coarse-to-fine pyramid** refines image and kernel across scales.

![DARSE-AC pipeline](assets/darse_ac_flow_diagram.png)

## Experimental Results

The method is evaluated on text images (ICDAR2019), natural scenes (iNaturalist 2021), low-light images (GLADNet), and realistic blur (RESID), covering parameter sensitivity, module ablation, convergence, and cross-dataset generalization.

| Dataset | PSNR | SSIM | ER | Success rate |
|---|--:|--:|--:|--:|
| iNaturalist 2021 | **31.03 dB** | **0.96** | **1.61** | **99.54%** |
| ICDAR2019 | 30.29 dB | 0.96 | — | — |
| GLADNet | 25.16 dB | 0.90 | — | — |
| RESID (total) | 20.56 dB | — | — | — |

**Key findings:**

- Salient-edge selection and AC regularization are complementary: jointly they restore text regions more clearly than either alone (Fig. 8).
- The objective curve of the improved algorithm flattens within about ten iterations; average kernel similarity increases with iterations (Fig. 9).
- The method achieves the best quantitative performance on iNaturalist 2021 compared with eight baselines, and shows strong cross-dataset generalization.
- Strong noise and spatially varying blur remain as open challenges.

![Paper Figure 8 – ICDAR2019 text ablation](assets/paper_fig8a_attributed_montage.png)

*Fig. 8 excerpt from the published paper: text-image restoration under parameter ablation (λ=0, θ=0, DARSE-AC). Full results and settings are described in the paper.*

## Application Context

The work was motivated by the need to restore blurred and noisy rubbing images related to the ancient Zhongshan Kingdom so that they become usable for design-material extraction. The paper establishes and validates the restoration algorithm on public datasets; application to actual cultural-object imagery requires authorized sources, expert evaluation of stroke fidelity, and systematic artifact review.

## Code

The implementation mirrors the paper's pipeline:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Synthetic evaluation and tests
.venv/bin/python experiment.py demo --output results/demo
.venv/bin/python experiment.py benchmark --output results/benchmark
.venv/bin/python experiment.py ablation --output results/ablation
.venv/bin/python -m pytest -q --junitxml=results/pytest.xml

# Restore a grayscale image
.venv/bin/python experiment.py deblur input.png \
  --output results/your_image --kernel-size 15 --levels 3 --outer 5
```

| File | Contents |
|---|---|
| `darse_ac.py` | Gradients/adjoints, periodic convolution, AC discretization, L0 thresholding, FFT image update, kernel estimation, pyramid, final restoration |
| `experiment.py` | `demo`, `benchmark`, `ablation`, `deblur` commands |
| `test_darse_ac.py` | Numerical operators, FFT normal equations, kernel normalization, determinism, CLI tests |
| `matlab/` | Toolbox-free MATLAB / Octave implementation of the same pipeline |
| `RECONSTRUCTION.md` | Equation-to-code conventions and discretization choices |
| `assets/` | Pipeline diagrams and attributed paper figures |

A matching **MATLAB / Octave** implementation lives in `matlab/` (see
`matlab/README.md`). It runs on R2016b+ and GNU Octave 6+ using core functions
only, and reproduces the same operators, HQS schedule, curvature surrogate, and
FFT updates.

Implementation conventions (grayscale `[0,1]`, centered odd kernels, periodic boundaries, HQS schedule, AC discrete surrogate) are documented in `RECONSTRUCTION.md`. Reported paper figures and metrics are quotes from the publication; figures remain with the authors and rights holders per the journal terms.

## Citation

```bibtex
@article{tan2025darseac,
  author  = {Tan, Shuo and Dong, Lingye and An, Limei},
  title   = {The Regularization and Deconvolution Algorithm Combining Salient Edges and Average Curvature in High-quality Visual Communication},
  journal = {Information Technology and Control},
  volume  = {54},
  number  = {1},
  pages   = {234--253},
  year    = {2025},
  doi     = {10.5755/j01.itc.54.1.38163}
}
```
