function [A, b_out, c, K] = sdpt3_to_sedumi(blk, At, C, b)
    blknum = size(blk,1);
    s = zeros(blknum,1);
    for k = 1:blknum
        if ~strcmp(blk{k,1}, 's')
            error('sdpt3_to_sedumi: only ''s'' blocks are supported.');
        end
        s(k) = blk{k,2};
    end
    m = length(b);
    cs2 = cumsum(s.^2);
    nvar = cs2(end);
    A      = sparse(m, nvar);
    c      = zeros(nvar, 1);
    b_out  = b;
    K.s    = s(:);   % only SDP blocks
    for k = 1:blknum
        sk = s(k);
        ck = C{k}(:);
        if k == 1
            idx = 1:cs2(k);
        else
            idx = cs2(k-1)+1:cs2(k);
        end
        c(idx) = ck;
        Ak_tri = At{k}';                 % m-by-ntri
        [m_check, ntri] = size(Ak_tri);
        if m_check ~= m || ntri ~= sk*(sk+1)/2
            error('sdpt3_to_sedumi: dimension mismatch in At{%d}.', k);
        end
        JJ1 = zeros(ntri,1);
        JJ2 = zeros(ntri,1);
        idx_tri = 0;
        for col = 1:sk
            for row = 1:col
                idx_tri = idx_tri + 1;
                JJ1(idx_tri) = row;
                JJ2(idx_tri) = col;
            end
        end
        [II, JJ, bb] = find(Ak_tri);     % 1-based indices
        rowIdx = JJ1(JJ);               % row index in sk-by-sk matrix
        colIdx = JJ2(JJ);               % col index in sk-by-sk matrix
        id_diag = find(rowIdx == colIdx);
        id_off  = find(rowIdx <  colIdx);
        I_diag = II(id_diag);
        J_diag = (colIdx(id_diag)-1)*sk + rowIdx(id_diag);  % vec index
        V_diag = bb(id_diag);
        I_off  = [II(id_off); II(id_off)];
        J_off  = [(colIdx(id_off)-1).*sk + rowIdx(id_off); ...
                  (rowIdx(id_off)-1).*sk + colIdx(id_off)];
        V_off  = [bb(id_off)/sqrt(2);    bb(id_off)/sqrt(2)];
        I_all  = [I_diag; I_off];
        J_all  = [J_diag; J_off];
        V_all  = [V_diag; V_off];
        Ak_full = sparse(I_all, J_all, V_all, m, sk*sk);
        A(:, idx) = Ak_full;
    end
end
