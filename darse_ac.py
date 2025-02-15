"""Author-maintained DARSE-AC reference implementation from the publication.

Reference: Tan, Dong, An (2025), DOI 10.5755/j01.itc.54.1.38163.
This is a clean reference release, not an archival copy of the experiment tree.
See RECONSTRUCTION.md for equation conventions and each disclosed substitution.
Arrays are float64 grayscale [0, 1]. Differences and convolution are periodic.
No training, learned weights, original datasets, or cultural-object images.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from skimage.transform import resize


@dataclass(frozen=True)
class Config:
    kernel_size: int = 15
    levels: int = 3
    outer: int = 5
    hqs_steps: int = 6
    edge: float = 4e-4
    l0: float = 4e-3
    curvature: float = 4e-3
    ridge: float = 2.0
    guide: float = 8e-4
    mu: float = 0.01
    penalty_start: float = 0.02
    penalty_growth: float = 2.0
    curvature_steps: int = 30
    guide_steps: int = 3
    final_reg: float = 0.002

    def validate(self):
        for name in ('kernel_size', 'levels', 'outer', 'hqs_steps',
                     'curvature_steps', 'guide_steps'):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.kernel_size < 3 or self.kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd and >= 3')
        for name, v in asdict(self).items():
            if not np.isfinite(v) or v < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if min(self.mu, self.penalty_start, self.ridge, self.final_reg) <= 0:
            raise ValueError('mu, penalty_start, ridge and final_reg must be positive')
        if self.penalty_growth < 1:
            raise ValueError('penalty_growth must be >= 1')


def image_array(image):
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 8:
        raise ValueError('expected a 2D grayscale image, each dimension >= 8')
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('image must be finite and in [0, 1]')
    return x.copy()


def grad(x):
    """Forward periodic differences, horizontal then vertical."""
    return np.roll(x, -1, axis=1) - x, np.roll(x, -1, axis=0) - x


def adjoint(gx, gy):
    """Exact adjoint D*, i.e. negative backward divergence."""
    return np.roll(gx, 1, axis=1) - gx + np.roll(gy, 1, axis=0) - gy


def normalize_kernel(k):
    k = np.asarray(k, dtype=float)
    if k.ndim != 2 or not np.isfinite(k).all():
        raise ValueError('kernel must be a finite 2D array')
    k = np.maximum(k, 0)
    if k.sum() <= 1e-15:
        k = np.zeros_like(k)
        k[k.shape[0] // 2, k.shape[1] // 2] = 1
    return k / k.sum()


def psf2otf(kernel, shape):
    """Centered spatial PSF -> OTF; explicit floor prevents odd-size shifts."""
    k = np.asarray(kernel, dtype=float)
    if k.ndim != 2 or any(a > b for a, b in zip(k.shape, shape)):
        raise ValueError('kernel must fit the image')
    padded = np.zeros(shape, dtype=float)
    padded[:k.shape[0], :k.shape[1]] = k
    for axis, size in enumerate(k.shape):
        padded = np.roll(padded, -(size // 2), axis=axis)
    return np.fft.fft2(padded)


def otf2psf(otf, shape):
    x = np.fft.ifft2(otf).real
    for axis, size in enumerate(shape):
        x = np.roll(x, size // 2, axis=axis)
    return x[:shape[0], :shape[1]].copy()


def convolve(x, k):
    return np.fft.ifft2(np.fft.fft2(x) * psf2otf(k, x.shape)).real


def gradient_power(shape):
    h, w = shape
    fx = np.exp(2j * np.pi * np.fft.fftfreq(w)) - 1
    fy = np.exp(2j * np.pi * np.fft.fftfreq(h)) - 1
    return np.abs(fx)[None, :] ** 2 + np.abs(fy)[:, None] ** 2


CURVATURE_STENCIL = np.array([[-1, 5, -1], [5, -16, 5], [-1, 5, -1]]) / 16


def curvature_filter(x):
    """Linear AC surrogate from paper Eq.(8), NOT exact graph mean curvature."""
    return convolve(x, CURVATURE_STENCIL)


def curvature_prox(image, strength, steps=30):
    """Solve .5||a-image||^2 + strength*||C a||_1 approximately.

    Dual projected gradient: a=image-C* p, |p|<=strength. C is symmetric.
    This reproducible convex surrogate replaces unspecified half-window filters.
    """
    if strength == 0:
        return image.copy()
    C = psf2otf(CURVATURE_STENCIL, image.shape)
    norm2 = float(np.max(np.abs(C) ** 2))
    dual = np.zeros_like(image)
    alpha = image.copy()
    for _ in range(steps):
        ca = np.fft.ifft2(C * np.fft.fft2(alpha)).real
        dual = np.clip(dual + ca / norm2, -strength, strength)
        alpha = image - np.fft.ifft2(np.conj(C) * np.fft.fft2(dual)).real
    # Safeguard the actual surrogate energy, not the ambiguous Eq.(9) sign.
    initial = strength * np.abs(curvature_filter(image)).sum()
    energy = .5 * np.square(alpha - image).sum() + strength * np.abs(curvature_filter(alpha)).sum()
    return alpha if energy <= initial + 1e-10 else image.copy()


def hard_threshold(gx, gy, weight, penalty):
    """Joint vector L0 prox: keep a pixel if ||gradient||^2 > xi/xi1."""
    mask = gx * gx + gy * gy > weight / penalty
    return gx * mask, gy * mask


def guided_structure(image, strength, mu=0.01, steps=3):
    """Frozen self-guidance approximation to Eq.(12).

    Each iteration freezes w=1/max(|D z|, mu) and solves
    (Id + strength D* w D) z = image via SciPy conjugate gradients.
    No external guide or hidden reference is used.
    """
    if strength == 0:
        return image.copy()
    z = image.copy()
    for _ in range(steps):
        gx, gy = grad(z)
        wx, wy = 1 / np.maximum(abs(gx), mu), 1 / np.maximum(abs(gy), mu)

        def matvec(v):
            a = v.reshape(image.shape)
            dx, dy = grad(a)
            return (a + strength * adjoint(wx * dx, wy * dy)).ravel()

        operator = LinearOperator((image.size, image.size), matvec=matvec, dtype=float)
        solution, info = cg(operator, image.ravel(), x0=z.ravel(), rtol=1e-6, maxiter=150)
        if info != 0:
            raise RuntimeError(f'guide CG did not converge: {info}')
        z = solution.reshape(image.shape)
    return z


def fft_image_update(observed, kernel, z, alpha, beta, edge, penalty, alpha_penalty):
    """Exact unconstrained quadratic minimizer of the image block of Eq.(6).

    The returned value is not clipped: clipping would invalidate its normal
    equation. Range clipping is applied only to the final display/output image.
    """
    K = psf2otf(kernel, observed.shape)
    power = gradient_power(observed.shape)
    rhs = np.conj(K) * np.fft.fft2(observed)
    rhs += edge * power * np.fft.fft2(z)
    rhs += penalty * np.fft.fft2(adjoint(*beta))
    rhs += alpha_penalty * np.fft.fft2(alpha)
    denominator = np.abs(K) ** 2 + (edge + penalty) * power + alpha_penalty
    return np.fft.ifft2(rhs / np.maximum(denominator, 1e-15)).real


def estimate_kernel(observed, z, size, ridge):
    """Gradient-domain Fourier ridge solution followed by support projection.

    Projection is a heuristic, not the exact constrained ridge minimizer.
    Ridge uses unnormalised FFT coefficients and unnormalised spatial sums.
    """
    zx, zy = (np.fft.fft2(a) for a in grad(z))
    ax, ay = (np.fft.fft2(a) for a in grad(observed))
    K = (np.conj(zx) * ax + np.conj(zy) * ay) / (abs(zx)**2 + abs(zy)**2 + ridge)
    return normalize_kernel(otf2psf(K, (size, size)))


def restore_nonblind(observed, kernel, regularization=0.002):
    """Final gradient-Tikhonov solve; a declared substitute for paper postprocess."""
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError('regularization must be finite and positive')
    K = psf2otf(kernel, observed.shape)
    denom = abs(K)**2 + regularization * gradient_power(observed.shape)
    x = np.fft.ifft2(np.conj(K) * np.fft.fft2(observed) / np.maximum(denom, 1e-12)).real
    return np.clip(x, 0, 1)


def objective(observed, image, kernel, z, cfg):
    """Diagnostic Eq.(5)-inspired energy, not a convergence certificate.

    z, scale and penalties change across blocks. Values at different scales
    should not be interpreted as one globally decreasing objective sequence.
    """
    gx, gy = grad(image)
    zx, zy = grad(z)
    return float(np.square(convolve(image, kernel) - observed).sum()
                 + cfg.edge * (np.square(gx-zx).sum() + np.square(gy-zy).sum())
                 + cfg.l0 * np.count_nonzero(gx*gx + gy*gy > 1e-12)
                 + cfg.curvature * abs(curvature_filter(image)).sum())


def deblur(observed, config=None):
    """Blind multiscale reconstruction. Returns final image, kernel, trace.

    The sharp ground truth and true kernel are deliberately absent from this
    interface. Initial kernel is a centered impulse; no metric-based selection.
    """
    cfg = config or Config()
    cfg.validate()
    observed = image_array(observed)
    if cfg.kernel_size >= min(observed.shape):
        raise ValueError('kernel_size must be smaller than the image')
    scales = [0.5 ** i for i in reversed(range(cfg.levels))]
    scales = [s for s in scales if s == 1 or min(observed.shape) * s >= 24]
    kernel = None
    image = None
    trace = []
    for level, scale in enumerate(scales):
        shape = tuple(max(8, round(v * scale)) for v in observed.shape)
        a = resize(observed, shape, anti_aliasing=True, preserve_range=True)
        size = max(3, int(round(cfg.kernel_size * scale)))
        size += 1 - size % 2
        size = min(size, cfg.kernel_size)
        if kernel is None:
            kernel = np.zeros((size, size))
            kernel[size // 2, size // 2] = 1
            image = a.copy()
        else:
            kernel = normalize_kernel(resize(kernel, (size, size), order=1,
                                             anti_aliasing=False, preserve_range=True))
            image = resize(image, shape, anti_aliasing=False, preserve_range=True)
        for iteration in range(cfg.outer):
            z = guided_structure(image, cfg.guide, cfg.mu, cfg.guide_steps)
            before = image.copy()
            for j in range(cfg.hqs_steps):
                penalty = cfg.penalty_start * cfg.penalty_growth ** j
                alpha_penalty = penalty if cfg.curvature > 0 else 0.0
                alpha = curvature_prox(image, cfg.curvature / (2 * penalty), cfg.curvature_steps)
                beta = hard_threshold(*grad(image), cfg.l0, penalty)
                image = fft_image_update(a, kernel, z, alpha, beta,
                                         cfg.edge, penalty, alpha_penalty)
            # Refresh the main structure before the kernel solve, as in alternation.
            z = guided_structure(image, cfg.guide, cfg.mu, cfg.guide_steps)
            kernel = estimate_kernel(a, z, size, cfg.ridge)
            trace.append({'level': level, 'scale': scale, 'iteration': iteration,
                          'height': shape[0], 'width': shape[1], 'kernel_size': size,
                          'energy_per_pixel': objective(a, image, kernel, z, cfg) / a.size,
                          'relative_change': float(np.linalg.norm(image-before) /
                                                   max(np.linalg.norm(before), 1e-12)),
                          'kernel_sum': float(kernel.sum()),
                          'kernel_min': float(kernel.min())})
    final = restore_nonblind(observed, kernel, cfg.final_reg)
    return final, kernel, trace
