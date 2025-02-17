%DARSE_AC_DEMO Synthetic blind-deblur demonstration for the MATLAB reference.
% Runs on Octave and MATLAB (R2016b+). Generates a small degraded image,
% restores it with darse_deblur, and writes PNGs + a metrics JSON to out_dir.
% Synthetic text/geometry only: NOT an artifact, NOT paper data.
clear; close all; clc;

out_dir = 'out';
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
rng(2025); % fixed seed, deterministic

size_ = 160;
truth = darse_synthetic(size_);
kernel = darse_true_kernel(15, 'motion');
observed = darse_convolve(truth, kernel);
observed = min(max(observed + 0.005 * randn(size_), 0), 1);

cfg = darse_config();
fprintf('Running DARSE-AC (MATLAB reference) on %dx%d, kernel %dx%d ...\n', ...
    size_, size_, cfg.kernel_size, cfg.kernel_size);

[restored, est_kernel, trace] = darse_deblur(observed, cfg);

psnr_ = 10 * log10(1 / mean((truth(:) - restored(:)).^2));
ssim_ = darse_ssim(truth, restored);
[gx, gy] = darse_grad(truth);
var_ = var(truth(:));
if var_ == 0; var_ = 1; end
fprintf('Restored: PSNR %.3f dB, SSIM %.4f\n', psnr_, ssim_);
fprintf('Kernel L1 error vs truth: %.4f\n', sum(abs(est_kernel(:) - kernel(:))));
fprintf('Final energy per pixel: %.4f\n', trace{end}.energy_per_pixel);

imwrite(truth, fullfile(out_dir, 'truth.png'));
imwrite(observed, fullfile(out_dir, 'observed.png'));
imwrite(restored, fullfile(out_dir, 'restored.png'));
imwrite(est_kernel ./ max(est_kernel(:)), fullfile(out_dir, 'kernel.png'));

metrics = struct('image', 'text_geometry', 'blur', 'motion', 'seed', 2025, ...
    'psnr_db', psnr_, 'ssim', ssim_, 'n_cases', 1, ...
    'config', cfg, 'note', 'Synthetic demonstration, not paper data.');
json_text = darse_metrics_json(metrics);
fid = fopen(fullfile(out_dir, 'metrics.json'), 'w');
fwrite(fid, json_text, 'char');
fclose(fid);
fprintf('Saved outputs to %s/\n', out_dir);
