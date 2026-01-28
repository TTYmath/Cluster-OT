function prob = sdpt3_to_mosek_multisdp(blk, At, C, b)
    nb = size(blk, 1);
    s_rows = find(strcmp(blk(:,1), 's'));

    if isempty(s_rows)
        error('sdpt3_to_mosek_multisdp: no ''s'' blocks found in blk.');
    end

    non_s = find(~strcmp(blk(:,1), 's'));
    if ~isempty(non_s)
        error(['sdpt3_to_mosek_multisdp: found non-''s'' block types. ', ...
               'This helper assumes a pure SDP (no linear variables).']);
    end
    K = numel(s_rows);
    m = length(b(:));
    bardim = zeros(1, K);
    for k = 1:K
        p = s_rows(k);
        diminfo = blk{p,2};

        if ~isscalar(diminfo)
            error(['sdpt3_to_mosek_multisdp: blk{%d,2} must be a scalar ', ...
                   '(one matrix per ''s'' row).'], p);
        end
        n = diminfo;
        bardim(k) = n;
        Ai = At{p};
        Ci = C{p};

        if size(Ai,1) ~= n*(n+1)/2 || size(Ai,2) ~= m
            error('At{%d} must be of size %d x %d.', p, n*(n+1)/2, m);
        end
        if ~isequal(size(Ci), [n, n])
            error('C{%d} must be %d-by-%d.', p, n, n);
        end
    end
    prob = struct();
    prob.sense = 'min';
    prob.c   = [];
    prob.a   = sparse(m, 0);   % m x 0
    prob.blx = [];
    prob.bux = [];
    prob.blc = b(:);
    prob.buc = b(:);
    prob.bardim = bardim;
    barc_subj = [];
    barc_subk = [];
    barc_subl = [];
    barc_val  = [];

    for k = 1:K
        p  = s_rows(k);
        n  = bardim(k);
        Ci = C{p};

        Csym = (Ci + Ci.')/2;            % enforce symmetry
        [Ik, Jk, Vk] = find(tril(Csym)); % lower-triangular entries

        barc_subj = [barc_subj; k * ones(numel(Vk),1)];
        barc_subk = [barc_subk; Ik];
        barc_subl = [barc_subl; Jk];
        barc_val  = [barc_val;  Vk];
    end

    prob.barc.subj = barc_subj;
    prob.barc.subk = barc_subk;
    prob.barc.subl = barc_subl;
    prob.barc.val  = barc_val;
    bara_subi = [];
    bara_subj = [];
    bara_subk = [];
    bara_subl = [];
    bara_val  = [];

    for k = 1:K
        p  = s_rows(k);
        n  = bardim(k);
        Ai = At{p};     % (n*(n+1)/2) x m

        for i = 1:m
            v = Ai(:, i);
            Aik_mat = smat_from_svec_sdpt3(v,n);
            [Ik, Jk, Vk] = find(tril(Aik_mat));
            ni = numel(Vk);

            bara_subi = [bara_subi; i * ones(ni,1)];  % constraint index
            bara_subj = [bara_subj; k * ones(ni,1)];  % PSD block index
            bara_subk = [bara_subk; Ik];
            bara_subl = [bara_subl; Jk];
            bara_val  = [bara_val;  Vk];
        end
    end

    prob.bara.subi = bara_subi;
    prob.bara.subj = bara_subj;
    prob.bara.subk = bara_subk;
    prob.bara.subl = bara_subl;
    prob.bara.val  = bara_val;
end

function X = smat_from_svec_sdpt3(v, n)
    [idx, ~, val] = find(v);   % idx: positions in v where v ~= 0

    if isempty(idx)
        X = sparse(n,n);
        return;
    end
    j = ceil( (sqrt(8*idx + 1) - 1) / 2 );
    t_prev = (j - 1) .* j / 2;    % T_{j-1}
    i = idx - t_prev;             % row index in [1..j]
    diagMask = (i == j);
    offMask  = ~diagMask;

    rt2 = sqrt(2);
    I_diag = i(diagMask);
    J_diag = j(diagMask);
    V_diag = val(diagMask);
    i_off = i(offMask);
    j_off = j(offMask);
    v_off = val(offMask) / rt2;
    I = [I_diag; i_off; j_off];
    J = [J_diag; j_off; i_off];
    V = [V_diag; v_off; v_off];

    X = sparse(I, J, V, n, n);
end

