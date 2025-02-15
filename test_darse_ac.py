"""Numerical correctness and interface tests, not paper-performance assertions."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scipy.signal import convolve2d
from darse_ac import (Config, grad, adjoint, psf2otf, otf2psf, convolve,
                      normalize_kernel, curvature_filter, curvature_prox,
                      hard_threshold, fft_image_update, estimate_kernel,
                      guided_structure, deblur, restore_nonblind)
from experiment import synthetic_image, true_kernel, metrics


def test_gradient_adjoint():
    rng = np.random.default_rng(2)
    x, gx, gy = rng.normal(size=(3, 23, 31))
    dx, dy = grad(x)
    np.testing.assert_allclose(np.sum(dx*gx+dy*gy), np.sum(x*adjoint(gx, gy)), atol=1e-12)


@pytest.mark.parametrize('size', [3, 5, 9])
def test_fft_matches_spatial_convolution(size):
    rng = np.random.default_rng(3)
    x = rng.random((27, 29))
    k = normalize_kernel(rng.random((size, size)))
    np.testing.assert_allclose(convolve(x, k), convolve2d(x, k, mode='same', boundary='wrap'), atol=1e-12)
    np.testing.assert_allclose(otf2psf(psf2otf(k, x.shape), k.shape), k, atol=1e-12)


def test_delta_kernel_no_shift():
    x = np.random.default_rng(4).random((24, 25))
    k = np.zeros((7, 7))
    k[3, 3] = 1
    np.testing.assert_allclose(convolve(x, k), x, atol=1e-12)


@pytest.mark.parametrize('value', [0, -1, 2])
def test_kernel_projection(value):
    k = normalize_kernel(np.full((5, 5), value))
    assert k.min() >= 0
    assert np.isclose(k.sum(), 1)


def test_curvature_constant_and_prox_energy():
    np.testing.assert_allclose(curvature_filter(np.ones((24, 25))), 0, atol=1e-12)
    x = np.random.default_rng(5).random((24, 25))
    strength = .05
    a = curvature_prox(x, strength, 80)
    new = .5*np.square(a-x).sum() + strength*abs(curvature_filter(a)).sum()
    assert new <= strength*abs(curvature_filter(x)).sum() + 1e-10
    np.testing.assert_array_equal(curvature_prox(x, 0), x)


def test_l0_threshold_minimizes_pixel_objective():
    gx = np.array([[.1, 2., 0.]])
    gy = np.array([[.1, 0., 1.]])
    bx, by = hard_threshold(gx, gy, 1., 1.)
    cost = ((bx != 0) | (by != 0)).astype(float) + (bx-gx)**2 + (by-gy)**2
    np.testing.assert_allclose(cost, np.minimum(1, gx*gx+gy*gy))


def test_fft_update_normal_equation():
    rng = np.random.default_rng(6)
    observed, z, alpha, bx, by = rng.random((5, 24, 25))
    k = normalize_kernel(rng.random((5, 5)))
    edge, penalty, ap = .1, .3, .2
    x = fft_image_update(observed, k, z, alpha, (bx, by), edge, penalty, ap)
    K = psf2otf(k, observed.shape)
    adj_data = np.fft.ifft2(np.conj(K)*np.fft.fft2(convolve(x, k)-observed)).real
    residual = adj_data + edge*adjoint(*grad(x-z)) + penalty*(adjoint(*grad(x))-adjoint(bx, by)) + ap*(x-alpha)
    assert np.max(abs(residual)) < 1e-11


def test_guidance_constant_and_identity():
    x = np.ones((24, 25))*.4
    np.testing.assert_allclose(guided_structure(x, .002), x, atol=1e-12)
    np.testing.assert_array_equal(guided_structure(x, 0), x)


def test_kernel_estimation_known_sharp():
    z = np.random.default_rng(7).random((48, 48))
    k = true_kernel(9, 'motion')
    estimated = estimate_kernel(convolve(z, k), z, 9, 1e-8)
    np.testing.assert_allclose(estimated, k, atol=1e-3)


def test_deterministic_blind_pipeline_and_invariants():
    x = synthetic_image(64)
    cfg = Config(kernel_size=7, levels=2, outer=2, hqs_steps=3, curvature_steps=8)
    a = convolve(x, true_kernel(7, 'motion'))
    outputs = [deblur(a, cfg) for _ in range(2)]
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    assert outputs[0][2] == outputs[1][2]
    y, k, trace = outputs[0]
    assert y.shape == x.shape and k.shape == (7, 7)
    assert np.isfinite(y).all() and 0 <= y.min() <= y.max() <= 1
    assert k.min() >= 0 and np.isclose(k.sum(), 1)
    assert len(trace) == 4
    assert {r['scale'] for r in trace} == {.5, 1}


def test_small_constant_input():
    x = np.full((12, 13), .4)
    y, k, trace = deblur(x, Config(kernel_size=3, outer=1, hqs_steps=2))
    np.testing.assert_allclose(y, x, atol=1e-12)
    assert len(trace) == 1


@pytest.mark.parametrize('field,value', [('kernel_size', 4), ('levels', 0), ('ridge', 0),
                                         ('l0', -1), ('mu', float('nan')), ('penalty_growth', .5)])
def test_invalid_configs(field, value):
    with pytest.raises(ValueError):
        replace(Config(), **{field: value}).validate()


@pytest.mark.parametrize('x', [np.ones((24, 24))*2, np.full((24, 24), np.nan), np.ones((24, 24, 3))])
def test_invalid_images(x):
    with pytest.raises(ValueError):
        deblur(x)


def test_oracle_restoration_improves_psnr_for_gaussian():
    x = synthetic_image(64)
    k = true_kernel(7, 'gaussian')
    a = convolve(x, k)
    result = restore_nonblind(a, k)
    assert np.mean((x-result)**2) < np.mean((x-a)**2)
    m = metrics(x, result, a, result)
    assert m['er_proxy'] == pytest.approx(1)


@pytest.mark.parametrize('field', ['edge', 'curvature', 'l0', 'guide'])
def test_ablations_run(field):
    cfg = Config(kernel_size=7, levels=1, outer=1, hqs_steps=2, curvature_steps=5)
    result, k, _ = deblur(synthetic_image(32), replace(cfg, **{field: 0}))
    assert np.isfinite(result).all() and np.isclose(k.sum(), 1)


def test_cli_demo_and_real_image(tmp_path):
    root = Path(__file__).parent
    process = subprocess.run([sys.executable, str(root/'experiment.py'), 'demo', '--size', '64',
                              '--kernel-size', '7', '--outer', '1', '--output', str(tmp_path/'demo')],
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    report = json.loads((tmp_path/'demo'/'summary.json').read_text())
    assert report['full']['n'] == 1
    input_path = next((tmp_path/'demo').glob('*/observed.png'))
    process = subprocess.run([sys.executable, str(root/'experiment.py'), 'deblur', str(input_path),
                              '--kernel-size', '7', '--outer', '1', '--output', str(tmp_path/'restore')],
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert (tmp_path/'restore'/'restored.npz').exists()
