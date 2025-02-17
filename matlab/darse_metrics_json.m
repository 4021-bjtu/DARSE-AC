function json = darse_metrics_json(metrics)
%DARSE_METRICS_JSON Minimal JSON serializer for the metrics struct (demo only).
% No external JSON library dependency; handles the fields used by the demo.
function out = esc(v)
    t = sprintf('%s', v);
    t = strrep(t, '\', '\\');
    t = strrep(t, '"', '\"');
    out = t;
end
lines = {};
lines{end + 1} = '{';
lines{end + 1} = sprintf('  "image": "%s",', esc(metrics.image));
lines{end + 1} = sprintf('  "blur": "%s",', esc(metrics.blur));
lines{end + 1} = sprintf('  "seed": %d,', metrics.seed);
lines{end + 1} = sprintf('  "psnr_db": %.6f,', metrics.psnr_db);
lines{end + 1} = sprintf('  "ssim": %.6f,', metrics.ssim);
lines{end + 1} = sprintf('  "n_cases": %d,', metrics.n_cases);
c = metrics.config;
lines{end + 1} = sprintf('  "config": {{"kernel_size": %d, "levels": %d, "outer": %d, "hqs_steps": %d, "edge": %.6g, "l0": %.6g, "curvature": %.6g, "ridge": %.6g, "guide": %.6g, "mu": %.6g, "penalty_start": %.6g, "penalty_growth": %.6g, "curvature_steps": %d, "guide_steps": %d, "final_reg": %.6g}},', ...
    c.kernel_size, c.levels, c.outer, c.hqs_steps, c.edge, c.l0, c.curvature, ...
    c.ridge, c.guide, c.mu, c.penalty_start, c.penalty_growth, c.curvature_steps, ...
    c.guide_steps, c.final_reg);
lines{end + 1} = sprintf('  "note": "%s"', esc(metrics.note));
lines{end + 1} = '}';
json = strjoin(lines, newline);
end
