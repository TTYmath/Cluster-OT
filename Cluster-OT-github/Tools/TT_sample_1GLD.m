function [samples, logZ] = TT_sample_1GLD(d,h, beta, lambda, L, dx, nsamples)
% GL1D_TT_SAMPLE  TT-conditional sampler for discretized 1-D Ginzburg–Landau.
%   Variables: x_1,...,x_d on a uniform grid in [-L, L], boundary x_0=x_{d+1}=0.
%   Energy (per site): (lambda1/2)*((x_i - x_{i-1})/h)^2 + (1/(4*lambda1))*(1 - x_i^2)^2
%   h = 1/(d+1).
%
% Inputs:
%   d         : number of interior sites
%   beta      : inverse temperature
%   lambda1   : parameter λ1
%   L         : half-interval size; domain is [-L, L]
%   dx     : spatial discretization stepsize
%   nsamples  : number of i.i.d. samples to draw
%   seed      : (optional) rng seed
%
% Outputs:
%   samples   : nsamples x d matrix of sampled states (values in [-L, L])
%   logZ      : log partition function (of the discretized model)

    rng('default');
    Ngrid = ceil(2*L/dx);
    %----- grid and constants -------------------------------------------------
    xgrid = linspace(-L, L, Ngrid).';    % Ngrid x 1
    V  = (1./(4*lambda)) * (1 - xgrid.^2).^2;      % on-site double-well
    phi = exp(-beta * V);                            % site factor φ(x) = exp(-β V(x))

    % Pair quadratic: (lambda1/2)*((x - y)/h)^2
    % Build the (Ngrid x Ngrid) Gaussian-like kernel K(y_prev, y_curr)
    % We'll store T = exp(-β*(λ1/2h^2)*(x - y)^2) * φ(y)  (column-scaled by φ for current site)
    XY2 = (xgrid.^2) + (xgrid.').^2 - 2*(xgrid*xgrid.');  % (x - y)^2 matrix
    a = beta * lambda / (2*h^2);
    K = exp(-a * XY2);                    % Ngrid x Ngrid
    T = K .* (ones(Ngrid,1) * phi.');     % multiply each column by φ(y)

    % Boundary vectors:
    % Left boundary x_0 = 0 enters at site 1 via ψ_1(0, x_1)
    left_vec  = exp(-a * (xgrid - 0).^2) .* phi;   % ψ_1(0, x_1) = exp(-a*(x1-0)^2)*φ(x1)
    % Right boundary x_{d+1} = 0 enters after site d via ψ_{d+1}(x_d, 0) = exp(-a*(0 - x_d)^2)
    right_vec = exp(-a * (0 - xgrid).^2);          % Ngrid x 1

    %----- backward TT messages b{i} : ℝ^{Ngrid} --------------------------------
    % Define terminal message b_{d+1}(x_d) = ψ_{d+1}(x_d, 0)
    b = cell(d+1,1);
    b{d+1} = right_vec;                 % size Ngrid x 1

    % For i=d,d-1,...,2:
    %   b{i}(x_{i-1}) = sum_{x_i} T(x_{i-1}, x_i) * b{i+1}(x_i)
    for i = d : -1 : 2
        b{i} = T * b{i+1};             % Ngrid x 1
    end

    % Special for i=1, incorporate left boundary explicitly later

    %----- partition function --------------------------------------------------
    % Z = sum_{x1} [ ψ_1(0,x1) * b{2}(x1) ]
    Z = sum(left_vec .* b{2});
    logZ = log(Z);

    %----- sampling: sequential TT conditionals --------------------------------
    % p(x1) ∝ ψ_1(0,x1) * b{2}(x1)
    p1 = left_vec .* b{2};
    p1 = p1 / sum(p1);
    idx1 = draw_from_discrete(p1, nsamples);
    samples = zeros(nsamples, d);
    samples(:,1) = xgrid(idx1);

    % For i=2,...,d:
    % p(x_i | x_{i-1}) ∝ T(x_{i-1}, x_i) * b{i+1}(x_i)
    for i = 2:d
        bx = b{i+1};                       % Ngrid x 1
        for n = 1:nsamples
            prev = samples(n,i-1);
            % find nearest grid index for prev (since prev is on the grid already)
            % (if you later allow non-grid conditioning, use interpolation)
            [~, prev_idx] = min(abs(xgrid - prev));
            un = T(prev_idx,:).' .* bx;    % Ngrid x 1, unnormalized
            p  = un / sum(un);
            idx = draw_from_discrete(p, 1);
            samples(n,i) = xgrid(idx);
        end
    end
end

%------------------------ helpers -------------------------------------------
function idx = draw_from_discrete(p, nsamp)
    cdf = cumsum(p(:));
    cdf = cdf / cdf(end);
    u   = rand(nsamp,1);
    idx = 1 + sum(u > cdf.', 2);
end
