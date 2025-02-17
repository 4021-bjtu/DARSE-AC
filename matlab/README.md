# MATLAB / Octave Reference Implementation

Toolbox-free `darse_*` functions implementing the DARSE-AC pipeline. Runs as-is
on **MATLAB R2016b+** and on **GNU Octave ≥ 6** (core functions only; no images
package required). The math conventions mirror the Python reference exactly.

## Usage

```matlab
run('darse_ac_demo.m')   % synthetic, degraded, restored, kernel + metrics.json
```

Manual call:

```matlab
cfg    = darse_config();                       % defaults from the paper
config = darse_convolve(truth, blur_kernel);   % synthetic degradation (demo)
[restored, est_kernel, trace] = darse_deblur(config, cfg);
```

Adjust any hyperparameter before running:

```matlab
cfg.levels = 4;     % pyramid levels (paper uses 4)
cfg.outer  = 10;    % alternations per level (paper uses 10)
cfg.edge   = 4e-4;  % lambda
cfg.l0     = 4e-3;  % xi
cfg.curvature = 4e-3; % theta
cfg.ridge     = 2.0;  % tau
cfg.guide     = 8e-4; % gamma
```

## Function map

| File | Purpose |
|---|---|
| `darse_config.m` | Default hyperparameter set |
| `darse_deblur.m` | Blind multi-scale deblurring entry point |
| `darse_grad.m` / `darse_adjoint.m` | Periodic forward differences and exact adjoint |
| `darse_psf2otf.m` / `darse_otf2psf.m` / `darse_convolve.m` | FFT convolution machinery |
| `darse_curvature_filter.m` | Eq.(8) linear AC stencil `[-1,5,-1;5,-16,5;-1,5,-1]/16` |
| `darse_curvature_prox.m` | AC surrogate proximal via dual projected gradient |
| `darse_hard_threshold.m` | Joint vector L0 prox |
| `darse_guided_structure.m` / `darse_pcg.m` / `darse_matvec2d.m` | Frozen-weight guidance + bundled CG |
| `darse_fft_image_update.m` | Closed-form image update (`|K|² + (λ+ξ₁)|D|² + θ₁`) |
| `darse_estimate_kernel.m` | Gradient-domain ridge kernel estimate + projection |
| `darse_restore_nonblind.m` | Final gradient-Tikhonov restoration |
| `darse_objective.m` | Diagnostic Eq.(5)-inspired energy |
| `darse_resize.m` / `darse_rotate.m` | Toolbox-free resize / rotate helpers |
| `darse_ssim.m` | Border-symmetric SSIM (demo evaluation) |
| `darse_synthetic.m` / `darse_true_kernel.m` | Synthetic text/geometry & blur kernels |
| `darse_metrics_json.m` | Minimal JSON writer for demo metadata |

## Verification

Verified on Octave 7.3 with a 160×160 synthetic text patch, 15×15 motion kernel,
and σ=0.005 noise:

```text
PSNR 19.125 dB   SSIM 0.7006   kernel L1 error 1.8462
```

The values are produced by the clean-room reference implementation on a
synthetic example and are **not** the paper's published numbers (31.03 dB etc.,
which come from public datasets and the original pipeline). See the repository
root README and `RECONSTRUCTION.md` for conventions and known differences.

## Notes

- Grayscale float arrays in `[0, 1]`, periodic boundaries, centered odd kernels.
- The demo writes to `matlab/out/` (`.gitignore`d); regenerate at will.
