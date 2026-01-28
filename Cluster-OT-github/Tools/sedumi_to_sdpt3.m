function [blk, At, C, b_out] = sedumi_to_sdpt3(A, b, c, K)
s = K.s;
blknum = length(s);
%cs = cumsum(s);
cs2 = cumsum(s.^2);
At = cell(blknum,1);
C = cell(blknum,1);
b_out = b;
m = length(b);
blk = cell(blknum,2);
for k = 1:blknum
    if k == 1
        Ak = A(:,1:cs2(k));
        ck = c(1:cs2(k));
        C{k} = reshape(ck,s(k),s(k));
    else
        Ak = A(:,cs2(k-1)+1:cs2(k));
        ck = c(cs2(k-1)+1:cs2(k));
        C{k} = reshape(ck,s(k),s(k));
    end
    sk = s(k);
    [II,JJ,bb] = find(Ak);
    JJ2 = ceil(JJ/sk);
    JJ1 = JJ-(JJ2-1)*sk;
    id1 = find(JJ1<JJ2);
    id2 = find(JJ1<=JJ2);
    bb(id1) = bb(id1)*sqrt(2);
    II = II(id2);
    %JJ = JJ(id2);
    JJ1 = JJ1(id2);
    JJ2 = JJ2(id2);
    JJ = JJ2.*(JJ2-1)/2+JJ1;
    bb = bb(id2);
    Ak = sparse(II,JJ,bb,m,sk*(sk+1)/2);
    At{k} = Ak';
    blk{k,1} = 's';
    blk{k,2} = sk;
end


