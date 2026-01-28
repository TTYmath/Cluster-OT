%%--------------------------------------------------------------------- 
% Cluster moment relaxation for OT
% G is the adjacency matrix of the correlation graph 
% I is cluster of variables
% samplex, sampley are samplings from two distributions
% n is relaxation degree
% fval is optimal value for cluster OT
% Tmap is transport plan (function handle)
% ttime is the running time for solving the SDP problem
%%---------------------------------------------------------------------
function [fval,Tmap,ttime] = OT_Moment_relax_sparse(G,I,samplex,sampley,n)
%% reset names of variables
G = G-diag(diag(G));
K = size(G,1);
if length(I)~=K
    error('\n dimension of graph and cluster number are not consistent');
end
Ne1 = I;
d = 0; 
for i = 1:K 
    d = max(d,max(Ne1{i})); 
end 
if size(samplex,2)~=d 
    error('\n sample dimension error'); 
end 
%% Construct high order clusters
for k = 1:K 
    Ne1{k} = sort(Ne1{k},'ascend'); 
end 
I = zeros(d,1); 
for k = 1:K 
    I(Ne1{k}) = I(Ne1{k})+1; 
end 
if sum(I)~=d||~isempty(find(I>1, 1))
    error('\n Ne is not a partition');
end
N1 = zeros(K,1);
for i = 1:K
    N1(i) = length(Ne1{i});
end
[I1,J1] = find(triu(G));
n2 = length(I1);
Ne2 = cell(n2,1);
for k = 1:n2
    Ne2{k} = sort([Ne1{I1(k)},Ne1{J1(k)}],'ascend');
end
N2 = zeros(n2,1);
for k = 1:n2
    N2(k) = length(Ne2{k});
end
%% build up SDP model

% monomial basis of each cluster; [x,y]

MOID = cell(K,1);
MON = zeros(K,1);
for k = 1:K
    aa = [];
    nk = N1(k);
    for i = 1:(n+1)^(2*nk)
        si = dec2base(i-1,n+1);
        si = si-'0';
        if length(si)<2*nk
            si = [zeros(1,2*nk-length(si)),si];
        end
        if sum(si)<=n
            aa = [aa;si];
        end
    end
    MOID{k} = aa;
    MON(k) = size(aa,1);
end

% index of each cluster
IDX = cell(K,1);
for k = 1:K
    idx = 1:MON(k);
    idx(2:end) = idx(2:end)+sum(MON(1:k-1)-1);
    IDX{k} = idx;
end

nn = 1+sum(MON-1); % matrix dimension

blk = cell(1,2);
blk{1,1} = 's';
blk{1,2} = nn;

%% construct A

II = []; JJ = []; bb = []; b = []; cnt = 1;
% add moment constraints in kth block

for k = 1:K
    nk = N1(k);
    for j = 1:MON(k)
        for i = 1:j
            si = MOID{k}(i,:);
            sj = MOID{k}(j,:);
            si1 = sum(si(1:nk)); si2 = sum(si(nk+1:2*nk));
            sj1 = sum(sj(1:nk)); sj2 = sum(sj(nk+1:2*nk));
            if (si2 == 0) && (sj2 == 0) % moments in xi, xj
                val = getmom(si(1:nk)+sj(1:nk),Ne1{k},samplex);
                idi = IDX{k}(i); 
                idj = IDX{k}(j);
                if idi < idj
                    II = [II;(idj-1)*idj/2+idi];
                    JJ = [JJ;cnt];
                    bb = [bb;sqrt(2)/2];
                    b = [b;val];
                elseif idi == idj
                    II = [II;(idj-1)*idj/2+idi];
                    JJ = [JJ;cnt];
                    bb = [bb;1];
                    b = [b;val];
                else
                    II = [II;(idi-1)*idi/2+idj];
                    JJ = [JJ;cnt];
                    bb = [bb;sqrt(2)/2];
                    b = [b;val];
                end
                cnt = cnt+1;
            elseif (si1 == 0) && (sj1 == 0) % moments in yi, yj
                val = getmom(si(nk+1:2*nk)+sj(nk+1:2*nk),Ne1{k},sampley);
                idi = IDX{k}(i); 
                idj = IDX{k}(j);
                if idi < idj
                    II = [II;(idj-1)*idj/2+idi];
                    JJ = [JJ;cnt];
                    bb = [bb;sqrt(2)/2];
                    b = [b;val];
                elseif idi == idj
                    II = [II;(idj-1)*idj/2+idi];
                    JJ = [JJ;cnt];
                    bb = [bb;1];
                    b = [b;val];
                else
                    II = [II;(idi-1)*idi/2+idj];
                    JJ = [JJ;cnt];
                    bb = [bb;sqrt(2)/2];
                    b = [b;val];
                end
                cnt = cnt+1;
            end
        end
    end
end


% add moment constraints in k1,k2th block

for k = 1:n2
    k1 = I1(k);
    k2 = J1(k);
    nk1 = N1(k1);
    nk2 = N1(k2);
    for i = 1:MON(k1)
        for j = 1:MON(k2)
            si = MOID{k1}(i,:);
            sj = MOID{k2}(j,:);
            si1 = sum(si(1:nk1)); si2 = sum(si(nk1+1:2*nk1));
            sj1 = sum(sj(1:nk2)); sj2 = sum(sj(nk2+1:2*nk2));
            if (si2 == 0) && (sj2 == 0) % moments in xi, xj
                val = getmom([si(1:nk1),sj(1:nk2)],[Ne1{k1},Ne1{k2}],samplex);
                idi = IDX{k1}(i); 
                idj = IDX{k2}(j);
                if idi < idj
                    II = [II;(idj-1)*idj/2+idi];
                    JJ = [JJ;cnt];
                    bb = [bb;sqrt(2)/2];
                    b = [b;val];
                elseif idi == idj
                    II = [II;(idj-1)*idj/2+idi];
                    JJ = [JJ;cnt];
                    bb = [bb;1];
                    b = [b;val];
                else
                    II = [II;(idi-1)*idi/2+idj];
                    JJ = [JJ;cnt];
                    bb = [bb;sqrt(2)/2];
                    b = [b;val];
                end
                cnt = cnt+1;
            elseif (si1 == 0) && (sj1 == 0) % moments in yi, yj
                val = getmom([si(nk1+1:2*nk1),sj(nk2+1:2*nk2)],[Ne1{k1},Ne1{k2}],sampley);
                idi = IDX{k1}(i); 
                idj = IDX{k2}(j); 
                if idi < idj
                    II = [II;(idj-1)*idj/2+idi];
                    JJ = [JJ;cnt];
                    bb = [bb;sqrt(2)/2];
                    b = [b;val];
                elseif idi == idj
                    II = [II;(idj-1)*idj/2+idi];
                    JJ = [JJ;cnt];
                    bb = [bb;1];
                    b = [b;val];
                else
                    II = [II;(idi-1)*idi/2+idj];
                    JJ = [JJ;cnt];
                    bb = [bb;sqrt(2)/2];
                    b = [b;val];
                end
                cnt = cnt+1;
            end
        end
    end
end

% add consistent constraints in kth block

for k = 1:K
    aa = [];
    aaid = [];
    for j = 1:MON(k)
        for i = 1:j
            si = MOID{k}(i,:);
            sj = MOID{k}(j,:);
            sij = si+sj;
            if ~isempty(aa)
                h = find(sum(abs(aa-sij),2)==0,1);
                if ~isempty(h)
                    i1 = aaid(h,1);
                    j1 = aaid(h,2);
                    idi = IDX{k}(i);
                    idj = IDX{k}(j);
                    idi1 = IDX{k}(i1);
                    idj1 = IDX{k}(j1);
                    if idi < idj
                        II = [II;(idj-1)*idj/2+idi];
                        JJ = [JJ;cnt];
                        bb = [bb;sqrt(2)/2];
                    elseif idi==idj
                        II = [II;(idj-1)*idj/2+idi];
                        JJ = [JJ;cnt];
                        bb = [bb;1];
                    else
                        II = [II;(idi-1)*idi/2+idj];
                        JJ = [JJ;cnt];
                        bb = [bb;sqrt(2)/2];
                    end
                    if idi1 < idj1
                        II = [II;(idj1-1)*idj1/2+idi1];
                        JJ = [JJ;cnt];
                        bb = [bb;-sqrt(2)/2];
                    elseif idi1==idj1
                        II = [II;(idj1-1)*idj1/2+idi1];
                        JJ = [JJ;cnt];
                        bb = [bb;-1];
                    else
                        II = [II;(idi1-1)*idi1/2+idj1];
                        JJ = [JJ;cnt];
                        bb = [bb;-sqrt(2)/2];
                    end
                    b = [b;0];
                    cnt = cnt+1;
                else
                    aa = [aa;sij];
                    aaid = [aaid;[i,j]];
                end
            else
                aa = [aa;sij];
                aaid = [aaid;[i,j]];
            end
        end
    end
end

At = {sparse(II,JJ,bb,nn*(nn+1)/2,length(b))};

% construct C

IIC = [];
JJC = [];
bbC = [];
NC = zeros(d,1);
for k = 1:d
    for i = 1:K
        if ismember(k,Ne1{i})
            NC(k) = i;
            break;
        end
    end
end

for ck = 1:d
    k = NC(ck);
    nk = N1(k);
    [~,h] = ismember(ck,Ne1{k});
    s = zeros(1,2*nk);
    sxx = s; sxx(:,h) = 2;
    syy = s; syy(:,nk+h) = 2;
    sxy = s; sxy(:,h) = 1; sxy(:,nk+h) = 1;
    isxx = 0;
    isyy = 0;
    isxy = 0;
    for j = 1:MON(k)
        for i = 1:j
            sij = MOID{k}(i,:)+MOID{k}(j,:);
            idi = IDX{k}(i);
            idj = IDX{k}(j);
            if norm(sij-sxx,1) == 0 && isxx == 0
                isxx = 1;
                if idi ~= idj
                    IIC = [IIC;idi;idj];
                    JJC = [JJC;idj;idi];
                    bbC = [bbC;1/2;1/2];
                else
                    IIC = [IIC;idi];
                    JJC = [JJC;idj];
                    bbC = [bbC;1];
                end
            elseif norm(sij-syy,1) == 0 && isyy == 0
                isyy = 1;
                if idi ~= idj
                    IIC = [IIC;idi;idj];
                    JJC = [JJC;idj;idi];
                    bbC = [bbC;1/2;1/2];
                else
                    IIC = [IIC;idi];
                    JJC = [JJC;idj];
                    bbC = [bbC;1];
                end
            elseif norm(sij-sxy,1) == 0 && isxy == 0
                isxy = 1;
                if idi ~= idj
                    IIC = [IIC;idi;idj];
                    JJC = [JJC;idj;idi];
                    bbC = [bbC;-1;-1];
                else
                    IIC = [IIC;idi];
                    JJC = [JJC;idj];
                    bbC = [bbC;-2];
                end
            end
        end
    end
end

C = {sparse(IIC,JJC,bbC,nn,nn)};


fprintf('\n original matrix size: %2d',nn);

%% use Mosek

AT = At{1};
AAT = AT'*AT;
[L,D,p] = ldl(AAT,'vector');
dd = diag(D);
id = find(dd>1e-10);
id = p(id);
AT = AT(:,id);
b = b(id);
At = {AT};


[blk,At,C,b,Cset] = Sparsify(blk,At,C,b);
prob = sdpt3_to_mosek_multisdp(blk, At, C, b);

param = struct( ...
    'MSK_DPAR_INTPNT_CO_TOL_PFEAS', 1e-8, ...
    'MSK_DPAR_INTPNT_CO_TOL_DFEAS', 1e-8, ...
    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP', 1e-7);



if exist('mosekopt','file') ~= 3 && exist('mosekopt','file') ~= 2
    error(sprintf([ ...
        'MOSEK not found on the MATLAB path.\n\n', ...
        'This part of the code requires the MOSEK optimization toolbox.\n', ...
        'Please:\n', ...
        '  1) Obtain MOSEK from: https://www.mosek.com/downloads/\n', ...
        '  2) Install it following their instructions\n', ...
        '  3) Make sure "mosekopt" is on your MATLAB path.\n' ...
        ]));
end


tic;
[r,res] = mosekopt('minimize echo(10)', prob,param);
ttime = toc;

bardim = prob.bardim;     % e.g. [n1, n2, ...]
bars   = res.sol.itr.bars;

K1 = numel(bardim);
S = cell(K1,1);
offset = 0;

for k = 1:K1
    n1 = bardim(k);
    L = n1*(n1+1)/2;              % length of lower-tri vector for this block
    vk = bars(offset + (1:L));  % extract this block
    offset = offset + L;
    Sk = zeros(n1);
    Sk(tril(true(n1))) = vk;     % fill lower triangle
    Sk = Sk + tril(Sk,-1).';    % symmetrize
    S{k} = Sk;                  % dual slack matrix for block k
end


fval = res.sol.itr.pobjval;
SS = sparse(nn,nn);
CK = length(Cset);
for k = 1:CK
    SS(Cset{k},Cset{k}) = SS(Cset{k},Cset{k})+sparse(S{k});
end
S = SS;




%% extract transport map
idx = 1;
for k = 1:K
    nk = N1(k);
    for i = 2:MON(k)
        si = MOID{k}(i,:);
        if norm(si(nk+1:2*nk),1) == 0
            idx = [idx;IDX{k}(i)];
        end
    end
end 
S = S(idx,idx);
MOID = cell(K,1);
MON = zeros(K,1);
for k = 1:K
    aa = [];
    nk = N1(k);
    for i = 1:(n+1)^nk 
        si = dec2base(i-1,n+1); 
        si = si-'0'; 
        if length(si)<nk
            si = [zeros(1,nk-length(si)),si]; 
        end
        if sum(si)<=n
            aa = [aa;si]; 
        end
    end
    MOID{k} = aa; 
    MON(k) = size(aa,1); 
end

% index of each cluster

IDX = cell(K,1);

for k = 1:K
    idx = 1:MON(k);
    idx(2:end) = idx(2:end)+sum(MON(1:k-1)-1);
    IDX{k} = idx;
end
nn = 1+sum(MON-1); % matrix dimension
Tmap = @(x)TTmap(x,S,d,K,nn,NC,Ne1,MON,MOID,IDX);

function y = TTmap(x,S,d,K,nn,NC,Ne1,MON,MOID,IDX)
s = x;
x = zeros(nn,1);
for k = 1:K
    for j = 1:MON(k)
        sj = MOID{k}(j,:);
        val = getmom(sj,Ne1{k},s);
        x(IDX{k}(j)) = val;
    end
end
x = x'*S;
s1 = zeros(1,d);
for ck = 1:d
    k = NC(ck);
    xk = x(IDX{k});
    xk1 = zeros(MON(k),1);
    [~,h] = ismember(ck,Ne1{k});
    for j = 1:MON(k)
        sj = MOID{k}(j,:);
        if sj(h) == 0
            xk1(j) = 0;
        else
            hh = sj(h);
            sj(h) = sj(h)-1;
            val = hh*getmom(sj,Ne1{k},s);
            xk1(j) = val;
        end
    end
    s1(1,ck) = xk*xk1;
end
y = s1;




function [val] = getmom(s,Ne,samp)
samp = samp(:,Ne);
val = mean(prod(samp.^s,2));

