"""Deterministic local demo, benchmark, ablation and grayscale image CLI."""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, rotate
from skimage import data, color, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from darse_ac import Config, deblur, convolve, normalize_kernel, restore_nonblind


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def save_image(path, array):
    Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255 + .5)).save(path)


def synthetic_image(size=160):
    """Generated English text and geometry. NOT a rubbing or cultural artifact."""
    canvas = Image.new('L', (size * 2, size * 2), 235)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, size // 5))
    small = ImageFont.load_default(size=max(10, size // 8))
    draw.text((size*.12, size*.12), 'TEXT 2026', font=font, fill=20)
    draw.text((size*.12, size*.48), 'SYNTHETIC', font=small, fill=45)
    draw.text((size*.12, size*.71), 'NOT AN ARTIFACT', font=small, fill=45)
    draw.rectangle((size*.16, size, size*.88, size*1.75), outline=25, width=5)
    draw.ellipse((size*1.12, size*1.04, size*1.78, size*1.7), outline=65, width=6)
    for i in range(5):
        y = size * (1.1 + i*.12)
        draw.line((size*.25, y, size*.75, y), fill=35, width=2+i%2)
    return np.asarray(canvas.resize((size, size), Image.Resampling.LANCZOS), dtype=float) / 255


def true_kernel(size, kind):
    k = np.zeros((size, size))
    mid = size // 2
    if kind == 'motion':
        k[mid, 2:-2] = 1
        k = rotate(k, 23, reshape=False, order=1)
    elif kind == 'gaussian':
        k[mid, mid] = 1
        k = gaussian_filter(k, 1.4)
    else:
        raise ValueError('unknown blur kind')
    return normalize_kernel(k)


def metrics(truth, restored, blurred, oracle, crop=0):
    if crop:
        truth, restored, blurred, oracle = (x[crop:-crop, crop:-crop] for x in (truth, restored, blurred, oracle))
    mse = float(np.mean((truth-restored)**2))
    oracle_mse = float(np.mean((truth-oracle)**2))
    baseline_mse = float(np.mean((truth-blurred)**2))
    return {'psnr_db': float(peak_signal_noise_ratio(truth, restored, data_range=1)),
            'ssim': float(structural_similarity(truth, restored, data_range=1)),
            'mse': mse, 'er_proxy': mse / max(oracle_mse, 1e-12),
            'mse_over_input': mse / max(baseline_mse, 1e-12)}


def provenance(args, cfg):
    return {'seed': args.seed, 'size': args.size, 'noise_std': args.noise,
            'config': asdict(cfg), 'python': platform.python_version(),
            'platform': platform.platform(),
            'versions': {x: importlib.metadata.version(x) for x in ('numpy', 'scipy', 'scikit-image', 'pillow')},
            'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ('darse_ac.py', 'experiment.py')},
            'boundary': 'periodic convolution, additive Gaussian noise, clipped to [0,1]',
            'evaluation': 'grayscale float64 [0,1]; full frame and kernel-radius cropped; no alignment or best-case selection',
            'er_proxy': 'MSE(restoration, truth) / MSE(same Tikhonov solver with true kernel, truth); NOT paper ER',
            'success_proxy': 'fraction with ER_proxy <= 2; user-defined, NOT paper success protocol',
            'scope': 'Local research reconstruction. No original paper data and no cultural artifact images.'}


def run_experiments(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(kernel_size=args.kernel_size, levels=args.levels, outer=args.outer)
    cfg.validate()
    if args.size < 64 or cfg.kernel_size >= args.size or args.noise < 0 or not np.isfinite(args.noise):
        raise ValueError('size >=64, kernel<size and finite noise>=0 required')
    images = {'text_geometry': synthetic_image(args.size)}
    if args.command in ('benchmark', 'ablation'):
        images['camera'] = resize(img_as_float(data.camera()), (args.size, args.size), anti_aliasing=True)
        images['coins'] = resize(img_as_float(data.coins()), (args.size, args.size), anti_aliasing=True)
    variants = {'full': cfg}
    if args.command == 'ablation':
        variants.update({'no_edge_term': replace(cfg, edge=0),
                         'no_curvature': replace(cfg, curvature=0),
                         'no_l0': replace(cfg, l0=0),
                         'no_guidance': replace(cfg, guide=0),
                         'single_scale': replace(cfg, levels=1)})
    rows = []
    seeds = [args.seed] if args.command == 'demo' else list(range(args.seed, args.seed+args.seeds))
    kinds = ['motion'] if args.command == 'demo' else ['motion', 'gaussian']
    for image_name, truth in images.items():
        for kind in kinds:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                kt = true_kernel(cfg.kernel_size, kind)
                observed = np.clip(convolve(truth, kt) + rng.normal(0, args.noise, truth.shape), 0, 1)
                oracle_start = time.perf_counter()
                oracle = restore_nonblind(observed, kt, cfg.final_reg)
                oracle_seconds = time.perf_counter() - oracle_start
                case = out / f'{image_name}_{kind}_seed{seed}'
                case.mkdir(exist_ok=True)
                save_image(case / 'truth.png', truth)
                save_image(case / 'observed.png', observed)
                np.savez_compressed(case / 'inputs.npz', truth=truth, observed=observed, true_kernel=kt)
                results = [('input', observed, None, [], 0.0), ('oracle_true_kernel', oracle, kt, [], oracle_seconds)]
                for name, variant in variants.items():
                    start = time.perf_counter()
                    result, kernel, trace = deblur(observed, variant)
                    results.append((name, result, kernel, trace, time.perf_counter()-start))
                for name, result, kernel, trace, seconds in results:
                    m = metrics(truth, result, observed, oracle)
                    cropped = metrics(truth, result, observed, oracle, cfg.kernel_size//2)
                    row = {'image': image_name, 'blur': kind, 'seed': seed, 'method': name,
                           **m, **{'crop_'+key: value for key, value in cropped.items()},
                           'runtime_seconds': seconds}
                    row['kernel_cosine'] = None if kernel is None else float(np.sum(kernel*kt) /
                                                                                         (np.linalg.norm(kernel)*np.linalg.norm(kt)))
                    rows.append(row)
                    save_image(case / f'{name}.png', result)
                    if kernel is not None:
                        np.savez_compressed(case / f'{name}.npz', restored=result, kernel=kernel)
                    if trace:
                        save_json(case / f'{name}_trace.json', trace)
                fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), layout='constrained')
                panels = [('Truth', truth), ('Blurred input', observed),
                          ('Reconstruction', results[2][1]), ('Oracle true kernel', oracle)]
                for ax, (label, arr) in zip(axes, panels):
                    ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(label, fontsize=11)
                    ax.axis('off')
                fig.suptitle(f'LOCAL RUN: {image_name} / {kind} / seed {seed} (NOT cultural artifact)', fontsize=10)
                fig.savefig(case / 'comparison.png', dpi=160)
                plt.close(fig)
    save_json(out / 'metadata.json', provenance(args, cfg))
    save_json(out / 'metrics.json', rows)
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in ['input', 'oracle_true_kernel', *variants]:
        group = [r for r in rows if r['method'] == name]
        summary[name] = {'n': len(group), **{key: float(np.mean([r[key] for r in group]))
                                          for key in ('psnr_db', 'ssim', 'er_proxy', 'runtime_seconds')},
                         'psnr_std_population': float(np.std([r['psnr_db'] for r in group])),
                         'success_proxy_percent': 100*sum(r['er_proxy'] <= 2 for r in group)/len(group)}
    save_json(out / 'summary.json', summary)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    for name in summary:
        ratios = np.sort([r['er_proxy'] for r in rows if r['method'] == name])
        ax.step(ratios, np.arange(1, len(ratios)+1)/len(ratios), where='post', label=name)
    ax.set(xlabel='ER_proxy threshold (not paper ER)', ylabel='Fraction <= threshold',
           title='Local synthetic degradation: cumulative success proxy')
    ax.legend(fontsize=8)
    fig.savefig(out / 'success_proxy.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('demo', 'benchmark', 'ablation'):
        p = sub.add_parser(name)
        p.add_argument('--output', default=f'results/{name}')
        p.add_argument('--seed', type=int, default=2026)
        p.add_argument('--seeds', type=int, default=2)
        p.add_argument('--size', type=int, default=160)
        p.add_argument('--noise', type=float, default=.005)
        p.add_argument('--kernel-size', type=int, default=15)
        p.add_argument('--levels', type=int, default=3)
        p.add_argument('--outer', type=int, default=5)
    p = sub.add_parser('deblur')
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kernel-size', type=int, default=15)
    p.add_argument('--levels', type=int, default=3)
    p.add_argument('--outer', type=int, default=5)
    args = parser.parse_args()
    if args.command == 'deblur':
        a = np.asarray(Image.open(args.input).convert('L'), dtype=float)/255
        cfg = Config(kernel_size=args.kernel_size, levels=args.levels, outer=args.outer)
        x, k, trace = deblur(a, cfg)
        args.output.mkdir(parents=True, exist_ok=True)
        save_image(args.output / 'restored.png', x)
        np.savez_compressed(args.output / 'restored.npz', restored=x, kernel=k)
        save_json(args.output / 'trace.json', trace)
        save_json(args.output / 'config.json', asdict(cfg))
        print('Saved grayscale reconstruction; no reference metrics without a ground truth.')
    else:
        if args.seeds < 1 or args.seed < 0:
            parser.error('seeds must be positive and seed nonnegative')
        run_experiments(args)


if __name__ == '__main__':
    main()
