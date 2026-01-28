function [samples, logZ] = TT_cross_Ising(N, beta, J, h, nsamples)
% ISING_TT_SAMPLE  Exact TT-based sampler for 1-D Ising with nearest-neighbor coupling.
%   [SAMPLES, LOGZ] = ISING_TT_SAMPLE(N, beta, J, h, nsamples, seed)
%   draws 'nsamples' independent configurations from
%       p(x) ∝ exp( beta*J*sum_{i=1}^{N-1} x_i x_{i+1} + beta*h*sum_{i=1}^N x_i ),
%   with x_i ∈ {-1,+1}. It uses TT/MPS-style right-to-left contraction
%   (backward messages) and left-to-right conditional sampling.
% 
% INPUTS
%   N         : number of spins
%   beta, J,h : model parameters (scalars)
%   nsamples  : how many i.i.d. samples to draw
% 
% OUTPUTS
%   samples   : nsamples x N matrix with entries in {-1,+1}
%   logZ      : log partition function (via exact TT contraction)
% 
% NOTES
%   • For 1-D Ising, the exact TT ranks are ≤ 2. This code performs the same
%     contractions you would use to evaluate/normalize a TT built by TT-cross,
%     but here no interpolation is necessary because the chain factorization
%     is explicit.
%   • Numerically stable (uses log-sum-exp where helpful).
% 
% EXAMPLE
%   [S, logZ] = ising_tt_sample(200, 0.7, 1.0, 0.2, 500, 1);
%   meanMag   = mean(mean(S,2));  % average magnetization over samples
    rng('default');

    % State space and indexing helpers
    states = [-1; +1];    % map index 1 -> -1, index 2 -> +1
    d = 2;                % local dimension

    % Site potentials φ_i(s) = exp(beta * h * s), same h for all i
    phi = exp(beta * h * states);  % 2x1

    % Pair potential ψ(s, t) = exp(beta * J * s * t) as a 2x2 "transfer matrix"
    % rows: s in {-1,+1}, cols: t in {-1,+1} with order states = [-1,+1]
    T = zeros(d, d);
    for i = 1:d
        for j = 1:d
            T(i,j) = exp(beta * J * states(i) * states(j));
        end
    end

    % ---------- Backward TT messages (right-to-left contraction) ----------
    % m_{i+1 -> i}(s_i) = sum_{s_{i+1}}  φ(s_{i+1}) * ψ(s_i, s_{i+1}) * m_{i+2->i+1}(s_{i+1})
    % with terminal message m_{N+1 -> N}(s_N) = 1  (no factor beyond N).
    b = cell(N,1);
    b{N} = ones(d,1);                      % m_{N+1->N}
    for i = N-1 : -1 : 1
        % Compute b{i}(s_i) = sum_{t} T(s_i,t) * [phi(t) .* b{i+1}(t)]
        tmp = phi .* b{i+1};               % elementwise for s_{i+1}
        b{i} = T * tmp;                    % size 2x1, indexed by s_i
    end

    % Partition function Z = sum_{s1} φ(s1) * b{1}(s1)
    Z = sum(phi .* b{1});
    logZ = log(Z);

    % ---------- Sampling: left-to-right using TT conditionals ----------
    % p(s1) ∝ φ(s1) * b{1}(s1)
    samples = zeros(nsamples, N);
    p1 = phi .* b{1};
    p1 = p1 / sum(p1);

    % Draw s1 for all samples
    s1_idx = draw_from_discrete(p1, nsamples);  % returns indices in {1,2}
    samples(:,1) = states(s1_idx);

    % For i >= 2: p(s_i | s_{i-1}) ∝ φ(s_i) * ψ(s_{i-1}, s_i) * b{i}(s_i)
    for i = 2:N
        si_idx = zeros(nsamples,1);
        for n = 1:nsamples
            prev = samples(n, i-1);
            prev_idx = 1 + (prev == +1);     % map -1 -> 1, +1 -> 2
            % unnormalized conditional over s_i ∈ {-1,+1}
            un = phi .* (T(prev_idx, :).') .* b{i};
            p  = un / sum(un);
            si_idx(n) = draw_from_discrete(p, 1);
        end
        samples(:,i) = states(si_idx);
    end
end

% -------------------------------------------------------------------------
function idx = draw_from_discrete(p, nsamp)
% Draw indices from a discrete distribution p over {1..length(p)}.
    cdf = cumsum(p(:)) / sum(p);
    u   = rand(nsamp,1);
    idx = 1 + sum(u > cdf.', 2);   % works for column cdf
end
