%% Marginal relaxtion for OT between Ising model 
%% bdw is bandwidth: bdw=0, meanfield 
%% Ne is the partition of [n]
function [fval,ttime] = Ising_MAR(G,Ne1,Gr,C1,c1,T1,C2,c2,T2)
[I,J,~] = find(triu(G)); 
m = length(I); 
n = size(G,1); 
r = length(c1{1}); 
if r^(2*n) > 1e8 
    error('\n too large'); 
end 
%% Diagonolization: get marginal distribution 
q1 = zeros(r^n,1); 
for i = 1:r^n
    % construct index
    s = dec2base(i-1,r);
    s = s-'0'; 
    if length(s)<n
        s = [zeros(1,n-length(s)),s]; 
    end
    s = s+1;
    for k = 1:n 
        q1(i) = q1(i)+c1{k}(s(k));
    end
    for k = 1:m
        ii = I(k);
        jj = J(k);
        q1(i) = q1(i)+C1{k}(s(ii),s(jj));
    end
end
q1 = exp(-q1/T1); 
q1 = q1/sum(q1); 

q2= zeros(r^n,1);
for i = 1:r^n
    % construct index
    s = dec2base(i-1,r);
    s = s-'0';
    if length(s)<n
        s = [zeros(1,n-length(s)),s];
    end
    s = s+1;
    for k = 1:n
        q2(i) = q2(i)+c2{k}(s(k));
    end
    for k = 1:m
        ii = I(k);
        jj = J(k);
        q2(i) = q2(i)+C2{k}(s(ii),s(jj));
    end
end 
q2 = exp(-q2/T2); 
q2 = q2/sum(q2); 

%% Construct graph for Ne 

n1 = length(Ne1);

for k = 1:n1
    Ne1{k} = sort(Ne1{k},'ascend');
end

I = zeros(n,1); 

for k = 1:n1
    I(Ne1{k}) = I(Ne1{k})+1;
end

if sum(I)~=n||~isempty(find(I>1, 1))
    error('\n Ne is not a partition');
end

N1 = zeros(n1,1);

for i = 1:n1
    N1(i) = length(Ne1{i});
end

G1 = zeros(n1,n1);

for i = 1:n1
    for j = 1:n1
        if Gr(i,j) > 0 %sum(G(Ne1{i},Ne1{j}),'all')>0
            G1(i,j) = 1;
            G1(j,i) = 1;
        end
    end
end

%G1 = ones(n1,n1);

G1 = G1-diag(diag(G1)); 

[I1,J1] = find(triu(G1));
n2 = length(I1);

Ne2 = cell(n2,1);

for k = 1:n2
    Ne2{k} = sort([Ne1{I1(k)},Ne1{J1(k)}],'ascend');
end

N2 = zeros(n2,1);

for k = 1:n2
    N2(k) = length(Ne2{k});
end

%% Build up LP model 

A1 = cell(n1,1); 
c1 = cell(n1,1); 

A2 = cell(n2,1);
c2 = cell(n2,1);
b = []; 

% add marginal constraint

for k = 1:n1
    nk = N1(k);
    for i = 1:n1
        ni = N1(i);
        if i == k
            qm1 = getmarginal(q1,Ne1{i},n,r);
            b = [b;qm1];
            A1{i} = [A1{i};kron(sparse(ones(1,r^nk)),speye(r^nk))];
            qm2 = getmarginal(q2,Ne1{i},n,r);
            b = [b;qm2];
            A1{i} = [A1{i};kron(speye(r^nk),sparse(ones(1,r^nk)))];
        else
            A1{i} = [A1{i};sparse(2*r^nk,r^(2*ni))];
        end
    end
    for i = 1:n2
        ni = N2(i);
        A2{i} = [A2{i};sparse(2*r^nk,r^(2*ni))];
    end
end



for k = 1:n2
    nk = N2(k);
    for i = 1:n2 
        ni = N2(i); 
        if i == k
            qm1 = getmarginal(q1,Ne2{i},n,r);
            b = [b;qm1];
            A2{i} = [A2{i};kron(sparse(ones(1,r^nk)),speye(r^nk))];
            qm2 = getmarginal(q2,Ne2{i},n,r);
            b = [b;qm2];
            A2{i} = [A2{i};kron(speye(r^nk),sparse(ones(1,r^nk)))];
        else
            A2{i} = [A2{i};sparse(2*r^nk,r^(2*ni))];
        end
    end
    for i = 1:n1
        ni = N1(i);
        A1{i} = [A1{i};sparse(2*r^nk,r^(2*ni))];
    end
end

% add consistent constraint

for k = 1:n2
    i1 = I1(k); j1 = J1(k);
    nk = N2(k);
    ni = N1(i1);
    nj = N1(j1);
    % left marginal
    for i = 1:n2
        if i == k
            [~,Ii] = ismember(Ne1{i1},Ne2{k});
            Ak = getA(nk,Ii,r);
            A2{i} = [A2{i};Ak];
        else
            A2{i} = [A2{i};sparse(r^(2*ni),r^(2*N2(i)))];
        end
    end
    for i = 1:n1
        if i == i1
            A1{i} = [A1{i};-speye(r^(2*ni))];
        else
            A1{i} = [A1{i};sparse(r^(2*ni),r^(2*N1(i)))];
        end
    end
    b = [b;zeros(r^(2*ni),1)];
    % right marginal
    for j = 1:n2
        if j == k
            [~,Ij] = ismember(Ne1{j1},Ne2{k});
            Ak = getA(nk,Ij,r);
            A2{j} = [A2{j};Ak];
        else
            A2{j} = [A2{j};sparse(r^(2*nj),r^(2*N2(j)))];
        end
    end
    for j = 1:n1
        if j == j1
            A1{j} = [A1{j};-speye(r^(2*nj))];
        else
            A1{j} = [A1{j};sparse(r^(2*nj),r^(2*N1(j)))];
        end
    end
    b = [b;zeros(r^(2*nj),1)];
end



% construct c

for k = 1:n2
    nk = N2(k);
    c2{k} = zeros(r^(2*nk),1);
end

for k = 1:n1
    nk = N1(k);
    for i = 1:r^nk
        si = dec2base(i-1,r);
        si = si-'0';
        if length(si)<nk
            si = [zeros(1,nk-length(si)),si];
        end
        si = 1-2*si;
        for j = 1:r^nk
            sj = dec2base(j-1,r);
            sj = sj-'0';
            if length(sj)<nk
                sj = [zeros(1,nk-length(sj)),sj];
            end
            sj = 1-2*sj;
            c1{k}(i,j) = norm(si-sj)^2;
        end
    end
    c1{k} = c1{k}(:);
end


% combine all blocks

cc = [];
AA = [];
for k = 1:n1
    cc = [cc;c1{k}];
    AA = [AA,A1{k}];
end

for k = 1:n2
    cc = [cc;c2{k}];
    AA = [AA,A2{k}];
end

nn = length(cc);

prob.c = cc;
prob.a = sparse(AA);
prob.blc = b;
prob.buc = b;
prob.blx = zeros(nn,1);
prob.bux = inf*ones(nn,1);


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
[r,res] = mosekopt('minimize',prob);
ttime = toc;
fval = res.sol.bas.pobjval;

function qm = getmarginal(q,Ni,n,r)
for k = 1:n
    if ismember(k,Ni)
        if k == 1
            aa = speye(r);
        else
            aa = kron(aa,speye(r));
        end
    else
        if k == 1
            aa = sparse(ones(1,r));
        else
            aa = kron(aa,sparse(ones(1,r)));
        end
    end
end
qm = aa*q;

function A = getA(ni,Ii,r)
for k = 1:ni
    if ismember(k,Ii)
        if k == 1
            aa = speye(r);
        else
            aa = kron(aa,speye(r));
        end
    else
        if k == 1
            aa = sparse(ones(1,r));
        else
            aa = kron(aa,sparse(ones(1,r)));
        end
    end
end
A = kron(aa,aa);










