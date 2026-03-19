function [blk,At,C,b,Cset] = Sparsify(blk,At,C,b)

    mergeMode  = 'sparsecolo';  % 'none' | 'sparsecolo' | 'cubic'
    mergeSigma = 0.75;          % overlap threshold for 'sparsecolo'
    minMergeSize = 2;           % do not merge 1x1 cliques aggressively
    useSymamd = true;           % use AMD before sparse Cholesky
    if ~iscell(blk) || size(blk,1) ~= 1 || size(blk,2) < 2
        error('Sparsify:InputFormat', ...
              'This implementation expects a single SDPT3 SDP block.');
    end
    if ~strcmp(blk{1,1}, 's')
        error('Sparsify:InputFormat', ...
              'Only semidefinite blocks are supported.');
    end
    if numel(blk{1,2}) ~= 1
        error('Sparsify:InputFormat', ...
              'Input must be a single SDP block. blk{1,2} should be scalar.');
    end
    if size(blk,2) >= 3 && ~isempty(blk{1,3})
        error('Sparsify:LowRankNotSupported', ...
              'Low-rank SDPT3 storage (blk{1,3}) is not supported here.');
    end
    if ~iscell(At) || numel(At) ~= 1 || ~iscell(C) || numel(C) ~= 1
        error('Sparsify:InputFormat', ...
              'Expected At and C to have one cell corresponding to the SDP block.');
    end

    n  = blk{1,2};
    m0 = length(b);
    AT0 = sparse(At{1});
    C0  = sparse(0.5*(C{1} + C{1}'));

    if size(AT0,1) ~= n*(n+1)/2
        error('Sparsify:AtSize', 'At{1} has inconsistent number of rows.');
    end
    if size(AT0,2) ~= m0
        error('Sparsify:AtSize', 'At{1} has inconsistent number of columns.');
    end
    if ~isequal(size(C0), [n n])
        error('Sparsify:CSize', 'C{1} must be an n-by-n matrix.');
    end
    [iiG,jjG] = svec_pairs(n);

    pat = spones(C0);
    nzRowsAT = find(sum(spones(AT0), 2) > 0);
    if ~isempty(nzRowsAT)
        pat = pat + sparse(iiG(nzRowsAT), jjG(nzRowsAT), 1, n, n);
        pat = pat + sparse(jjG(nzRowsAT), iiG(nzRowsAT), 1, n, n);
    end
    pat = spones(pat);
    pat = pat + speye(n);  % make sure every vertex is present
    if useSymamd
        p = symamd(pat);
        if isempty(p)
            p = 1:n;
        end
    else
        p = 1:n;
    end

    Hp = spones(pat(p,p));
    Hp = tril(Hp,-1) + tril(Hp,-1)' + speye(n);

    % Add a large diagonal so the matrix is numerically SPD but the pattern
    % remains unchanged.
    L = chol(Hp + n*speye(n), 'lower');

    cand = cell(n,1);
    for j = 1:n
        idx = find(L(:,j));
        cand{j} = sort(p(idx));
    end
    Cset = maximalize_cliques(cand);

    if isempty(Cset)
        Cset = {1:n};
    end

    switch lower(mergeMode)
        case 'none'
            % no action
        case 'sparsecolo'
            Cset = merge_cliques_overlap(Cset, mergeSigma, minMergeSize);
        case 'cubic'
            Cset = merge_cliques_cubic(Cset, minMergeSize);
        otherwise
            error('Sparsify:MergeMode', 'Unknown mergeMode "%s".', mergeMode);
    end
    Cset = maximalize_cliques(Cset);

    % Sort for reproducibility
    Cset = sort_cliques(Cset);

    nblk = numel(Cset);
    blksz = zeros(1,nblk);
    loc   = cell(nblk,1);
    for k = 1:nblk
        Cset{k} = sort(unique(Cset{k}));
        blksz(k) = numel(Cset{k});
        lk = zeros(1,n);
        lk(Cset{k}) = 1:blksz(k);
        loc{k} = lk;
    end
    treeEdges = clique_tree(Cset);
    supportRows = false(n*(n+1)/2,1);
    supportRows(nzRowsAT) = true;

    [ic,jc] = find(tril(spones(C0)));
    if ~isempty(ic)
        supportRows(ltind_vec(ic,jc)) = true;
    end

    ownerRow = zeros(n*(n+1)/2,1);
    ownedRows = find(supportRows);
    for t = 1:length(ownedRows)
        r = ownedRows(t);
        ownerRow(r) = find_owner(iiG(r), jjG(r), loc, blksz);
    end

    barlen = sum(blksz .* (blksz + 1) / 2);
    offs   = [0, cumsum(blksz .* (blksz + 1) / 2)];

    [rA,cA,vA] = find(AT0);
    I0 = zeros(length(vA),1);
    for t = 1:length(vA)
        r = rA(t);
        ii = iiG(r);
        jj = jjG(r);
        k = ownerRow(r);
        if k == 0 || loc{k}(ii) == 0 || loc{k}(jj) == 0
            k = find_owner(ii, jj, loc, blksz);
        end
        li = loc{k}(ii);
        lj = loc{k}(jj);
        if li == 0 || lj == 0
            error('Sparsify:LocalIndex', ...
                  'Internal owner mismatch for At entry (%d,%d).', ii, jj);
        end
        if li > lj
            tmp = li; li = lj; lj = tmp;
        end
        I0(t) = offs(k) + ltind(li, lj);
    end

    Cblk = cell(nblk,1);
    for k = 1:nblk
        Cblk{k} = sparse(blksz(k), blksz(k));
    end

    % Work with the upper-triangular / canonical representative of each
    % symmetric entry so the packed row index matches ltind() exactly.
    [iC,jC,vC] = find(triu(C0));
    for t = 1:length(vC)
        ii = iC(t);
        jj = jC(t);
        r = ltind(ii, jj);
        k = ownerRow(r);
        if k == 0
            k = find_owner(ii, jj, loc, blksz);
        end
        li = loc{k}(ii);
        lj = loc{k}(jj);
        if li == 0 || lj == 0
            % Be defensive in case an owner was cached before clique merging
            % or due to any unexpected pattern mismatch.
            k = find_owner(ii, jj, loc, blksz);
            li = loc{k}(ii);
            lj = loc{k}(jj);
        end
        if li == 0 || lj == 0
            error('Sparsify:LocalIndexMismatch', ...
                  ['Failed to place the canonical C-entry (%d,%d). ' ...
                   'Owner clique %d does not contain both indices.'], ...
                  ii, jj, k);
        end
        Cblk{k}(li,lj) = Cblk{k}(li,lj) + vC(t);
        if li ~= lj
            Cblk{k}(lj,li) = Cblk{k}(lj,li) + vC(t);
        end
    end

    neq = 0;
    for e = 1:size(treeEdges,1)
        a = treeEdges(e,1);
        d = treeEdges(e,2);
        Sep = intersect(Cset{a}, Cset{d});
        s = numel(Sep);
        neq = neq + s*(s+1)/2;
    end

    Ieq = zeros(2*neq,1);
    Jeq = zeros(2*neq,1);
    Veq = zeros(2*neq,1);

    cntEq = 0;
    ptr = 0;
    for e = 1:size(treeEdges,1)
        a = treeEdges(e,1);
        d = treeEdges(e,2);
        Sep = intersect(Cset{a}, Cset{d});
        s = numel(Sep);

        for j = 1:s
            for i = 1:j
                u = Sep(i);
                v = Sep(j);

                la1 = loc{a}(u); la2 = loc{a}(v);
                ld1 = loc{d}(u); ld2 = loc{d}(v);

                if la1 > la2
                    tmp = la1; la1 = la2; la2 = tmp;
                end
                if ld1 > ld2
                    tmp = ld1; ld1 = ld2; ld2 = tmp;
                end

                cntEq = cntEq + 1;
                col = m0 + cntEq;

                if u == v
                    coef = 1.0;
                else
                    coef = sqrt(2)/2;
                end

                ptr = ptr + 1;
                Ieq(ptr) = offs(a) + ltind(la1, la2);
                Jeq(ptr) = col;
                Veq(ptr) =  coef;

                ptr = ptr + 1;
                Ieq(ptr) = offs(d) + ltind(ld1, ld2);
                Jeq(ptr) = col;
                Veq(ptr) = -coef;
            end
        end
    end

    %--------------------------------------------------------------
    % SDPT3 data
    %--------------------------------------------------------------
    % SDPT3 expects one row in blk per cone block.  Although SDPT3 also
    % allows a *single* row with many SDP subblocks packed together,
    % your downstream code expects the SparseCoLO / sedumi_to_sdpt3 style:
    %   blk : [nblk x 2] cell, one row per local PSD block,
    %   At  : [nblk x 1] cell,
    %   C   : [nblk x 1] cell.
    At_stack = sparse([I0; Ieq], [cA; Jeq], [vA; Veq], barlen, m0 + neq);

    blk = cell(nblk,2);
    At  = cell(nblk,1);
    C   = cell(nblk,1);
    for k = 1:nblk
        blk{k,1} = 's';
        blk{k,2} = blksz(k);
        rows = (offs(k)+1):offs(k+1);
        At{k,1} = At_stack(rows,:);
        C{k,1}  = Cblk{k};
    end
    b = [b(:); zeros(neq,1)];
end




function [ii,jj] = svec_pairs(n)

    jj = repelem((1:n)', (1:n)');
    nn = n*(n+1)/2;

    % Construct ii using cumulative structure
    ii = zeros(nn,1);
    idx = cumsum([1; (1:n-1)']);  % starting positions of each block

    for j = 1:n
        ii(idx(j):idx(j)+j-1) = (1:j)';
    end
end

function r = ltind(i,j)
% Lower-triangular index in SDPT3 ordering.
% The pair is canonicalized internally so ltind(i,j) = ltind(j,i).
    swap = i > j;
    if any(swap)
        tmp = i(swap);
        i(swap) = j(swap);
        j(swap) = tmp;
    end
    r = (j-1).*j/2 + i;
end

function r = ltind_vec(i,j)
% Vectorized lower-triangular index in SDPT3 ordering.
    r = ltind(i,j);
end

function C = maximalize_cliques(Cin)
% Remove empties, duplicates, and non-maximal subsets.
    tmp = {};
    for k = 1:numel(Cin)
        ck = unique(sort(Cin{k}));
        if ~isempty(ck)
            tmp{end+1} = ck; %#ok<AGROW>
        end
    end
    if isempty(tmp)
        C = {};
        return;
    end

    sz = cellfun(@numel, tmp);
    [~,ord] = sort(sz, 'descend');
    tmp = tmp(ord);

    C = {};
    for k = 1:numel(tmp)
        ck = tmp{k};
        isSubset = false;
        for j = 1:numel(C)
            if all(ismember(ck, C{j}))
                isSubset = true;
                break;
            end
        end
        if ~isSubset
            C{end+1} = ck; %#ok<AGROW>
        end
    end
end

function C = sort_cliques(C)
% Stable ordering for reproducibility.
    if isempty(C)
        return;
    end
    a = zeros(numel(C),1);
    s = zeros(numel(C),1);
    for k = 1:numel(C)
        a(k) = C{k}(1);
        s(k) = numel(C{k});
    end
    T = [(1:numel(C))' a s];
    T = sortrows(T, [2 3 1]);
    C = C(T(:,1));
end

function kbest = find_owner(i,j,loc,blksz)
% Choose the smallest clique containing both (i,j).
    kbest = 0;
    bestsz = inf;
    for k = 1:numel(loc)
        if loc{k}(i) > 0 && loc{k}(j) > 0
            if blksz(k) < bestsz
                kbest = k;
                bestsz = blksz(k);
            end
        end
    end
    if kbest == 0
        error('Sparsify:OwnerNotFound', ...
              'No clique contains entry (%d,%d).', i, j);
    end
end

function C = merge_cliques_overlap(C, sigma, minsz)
    if nargin < 3
        minsz = 2;
    end
    changed = true;
    while changed
        changed = false;
        bestPair = [0 0];
        bestScore = -inf;
        bestUnion = inf;

        nC = numel(C);
        for i = 1:nC-1
            Ci = C{i};
            if numel(Ci) < minsz, continue; end
            for j = i+1:nC
                Cj = C{j};
                if numel(Cj) < minsz, continue; end
                ov = numel(intersect(Ci,Cj));
                if ov == 0, continue; end
                score = min(ov/numel(Ci), ov/numel(Cj)) - sigma;
                unionsz = numel(union(Ci,Cj));
                if (score > bestScore + 1e-14) || ...
                   (abs(score - bestScore) <= 1e-14 && unionsz < bestUnion)
                    bestScore = score;
                    bestPair = [i j];
                    bestUnion = unionsz;
                end
            end
        end

        if bestScore >= 0 && all(bestPair > 0)
            i = bestPair(1);
            j = bestPair(2);
            Cm = union(C{i}, C{j});
            keep = true(1,numel(C));
            keep([i j]) = false;
            C = C(keep);
            C{end+1} = Cm;
            C = maximalize_cliques(C);
            changed = true;
        end
    end
end

function C = merge_cliques_cubic(C, minsz)
    if nargin < 2
        minsz = 2;
    end
    changed = true;
    while changed
        changed = false;
        bestPair = [0 0];
        bestScore = -inf;

        nC = numel(C);
        for i = 1:nC-1
            Ci = C{i};
            if numel(Ci) < minsz, continue; end
            si = numel(Ci);
            for j = i+1:nC
                Cj = C{j};
                if numel(Cj) < minsz, continue; end
                if isempty(intersect(Ci,Cj)), continue; end
                sj = numel(Cj);
                su = numel(union(Ci,Cj));
                score = si^3 + sj^3 - su^3;
                if score > bestScore
                    bestScore = score;
                    bestPair = [i j];
                end
            end
        end

        if bestScore > 0 && all(bestPair > 0)
            i = bestPair(1);
            j = bestPair(2);
            Cm = union(C{i}, C{j});
            keep = true(1,numel(C));
            keep([i j]) = false;
            C = C(keep);
            C{end+1} = Cm;
            C = maximalize_cliques(C);
            changed = true;
        end
    end
end

function E = clique_tree(C)
    nC = numel(C);
    if nC <= 1
        E = zeros(0,2);
        return;
    end

    W = zeros(0,3);
    for i = 1:nC-1
        for j = i+1:nC
            wij = numel(intersect(C{i}, C{j}));
            if wij > 0
                W(end+1,:) = [i j wij];
            end
        end
    end
    if isempty(W)
        E = zeros(0,2);
        return;
    end

    [~,ord] = sort(W(:,3), 'descend');
    W = W(ord,:);

    parent = 1:nC;
    rankv = zeros(1,nC);
    E = zeros(0,2);

    for t = 1:size(W,1)
        i = W(t,1);
        j = W(t,2);
        ri = uf_find(parent, i);
        rj = uf_find(parent, j);
        if ri ~= rj
            E(end+1,:) = [i j]; 
            [parent,rankv] = uf_union(parent, rankv, ri, rj);
        end
    end
end

function r = uf_find(parent, x)
    r = x;
    while parent(r) ~= r
        r = parent(r);
    end
end

function [parent,rankv] = uf_union(parent, rankv, a, b)
    if rankv(a) < rankv(b)
        parent(a) = b;
    elseif rankv(a) > rankv(b)
        parent(b) = a;
    else
        parent(b) = a;
        rankv(a) = rankv(a) + 1;
    end
end

function M = blkdiag_sparse(Cblk)
    if isempty(Cblk)
        M = sparse(0,0);
        return;
    end
    M = sparse(Cblk{1});
    for k = 2:numel(Cblk)
        M = blkdiag(M, sparse(Cblk{k}));
    end
    M = sparse(M);
end






