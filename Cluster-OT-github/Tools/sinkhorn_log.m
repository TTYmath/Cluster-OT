function [P, u, v, info] = sinkhorn_log(C, a, b, T, maxiter, tol)
    [n, m] = size(C);
    a = a(:); b = b(:);
    if length(a) ~= n || length(b) ~= m
        error('Dimensions of a or b are incompatible with C.');
    end
    if any(a < 0) || any(b < 0)
        error('Marginals a and b must be nonnegative.');
    end
    sa = sum(a); sb = sum(b);
    if sa <= 0 || sb <= 0
        error('Marginals must have positive total mass.');
    end
    a = a / sa;
    b = b / sb;
    if any(a == 0) || any(b == 0)
        error('This implementation assumes strictly positive marginals a, b.');
    end
    loga = log(a);
    logb = log(b);
    logK = -C / T;
    logu = zeros(n, 1);
    logv = zeros(m, 1);
    converged = false;
    err = inf;

    for it = 1:maxiter
        Mrow = logK + ones(n,1) * logv.';      % n x m
        row_lse = logsumexp(Mrow, 2);          % n x 1
        logu = loga - row_lse;
        Mcol = logK + logu * ones(1, m);       % n x m
        col_lse = logsumexp(Mcol, 1).';        % m x 1
        logv = logb - col_lse;
        logP = logu + logv.' + logK;   % n x m
        log_row = logsumexp(logP, 2);  % n x 1
        row = exp(log_row);
        log_col = logsumexp(logP, 1).'; % m x 1
        col = exp(log_col);

        err = (norm(row - a)+norm(col - b))/(1+norm(a)+norm(b));

        if err < tol
            converged = true;
            break;
        end
        fprintf('\n iter: %2d |  err: %2d ',it,err);

    end
    if ~exist('logP', 'var')
        logP = logu + logv.' + logK;
    end
    P = exp(logP);

    u = exp(logu);
    v = exp(logv);
    info.niter = it;
    info.err = err;
    info.converged = converged;
end
function y = logsumexp(X, dim)
    if nargin < 2
        dim = 1;
    end
    M = max(X, [], dim);
    M_safe = M;
    M_safe(~isfinite(M_safe)) = 0;
    Y = bsxfun(@minus, X, M_safe);
    S = sum(exp(Y), dim);
    y = M_safe + log(S);
    y(S == 0) = -Inf;
end
