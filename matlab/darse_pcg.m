function x = darse_pcg(matvec, b, x0, rtol, maxit)
%DARSE_PCG Minimal conjugate-gradient solver for SPD systems A x = b where
% A is given only through the function handle matvec(v).
% Returns the (possibly not fully converged) iterate; used by the frozen-weight
% guidance solve, mirroring scipy.sparse.linalg.cg in the Python reference.
b = b(:);
x = x0(:);
r = b - matvec(x);
p = r;
rr = r' * r;
if rr <= 0
    x = x0(:);
    return;
end
for k = 1:maxit
    Ap = matvec(p);
    alpha = (rr) / real(p' * Ap);
    x = x + alpha * p;
    r = r - alpha * Ap;
    rrn = r' * r;
    if sqrt(rrn / max(real(b' * b), 1e-30)) < rtol
        break;
    end
    beta = rrn / rr;
    p = r + beta * p;
    rr = rrn;
end
end
