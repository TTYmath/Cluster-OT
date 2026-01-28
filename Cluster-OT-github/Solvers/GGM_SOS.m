%% Cluster moment of OT between Gaussian distributions
%% G is the correlative graph
%% S1, S2 are covariance matrices of source and target distributions
%% mu1, mu2 are mean of source and target distributions
function [fval,ttime] = GGM_SOS(G,S1,S2,mu1,mu2)
fprintf('\n Basis: First order Moment'); 
G = abs(G)>0;
G = (G+G')/2;
G = G-diag(diag(G));
n = size(G,1);
m = nnz(G)/2;
[II,JJ,~] = find(triu(G));
blk{1,1} = 's'; blk{1,2} = 2*n+1;
I = zeros(4*n+2*m+1,1); J = zeros(4*n+2*m+1,1); b = zeros(4*n+2*m+1,1);
for i = 1:2*n
    I(i) = (i+1)*(i+2)/2;
    J(i) = i;
    b(i) = 1;
end
for i = 1:2*n
    I(2*n+i) = i*(i+1)/2+1;
    J(2*n+i) = 2*n+i;
    b(2*n+i) = sqrt(2)/2;
end
for i = 1:m
    x = II(i);
    y = JJ(i);
    I(4*n+i) = (1+y)*y/2+x+1;
    J(4*n+i) = 4*n+i;
    b(4*n+i) = sqrt(2)/2;
end
for i = 1:m
    x = II(i)+n;
    y = JJ(i)+n;
    I(4*n+m+i) = (1+y)*y/2+x+1;
    J(4*n+m+i) = 4*n+m+i;
    b(4*n+m+i) = sqrt(2)/2;
end
I(4*n+2*m+1) = 1;
J(4*n+2*m+1) = 4*n+2*m+1;
b(4*n+2*m+1) = 1;
At = {sparse(I,J,b,(2*n+1)*(n+1),4*n+2*m+1)};
C = speye(2*n+1);
C(2:n+1,n+2:2*n+1) = -speye(n);
C(n+2:2*n+1,2:n+1) = -speye(n);
C(1,1) = 0;
C = {C};
b = zeros(4*n+2*m+1,1);
b(1:2*n) = [mu1;mu2].^2+[diag(S1);diag(S2)];
b(2*n+1:4*n) = [mu1;mu2];
M1 = S1+mu1*mu1';
m1 = M1(:);
b(4*n+1:4*n+m) = m1(II+(JJ-1)*n);
M2 = S2+mu2*mu2';
m2 = M2(:);
b(4*n+m+1:4*n+2*m) = m2(II+(JJ-1)*n);
b(4*n+2*m+1) = 1;


% Choldal conversion

[blk,At,C,b,CSet] = Sparsify(blk,At,C,b);

prob = sdpt3_to_mosek_multisdp(blk, At, C, b);

tol = 1e-12;

param = struct( ...
    'MSK_DPAR_INTPNT_CO_TOL_PFEAS', tol, ...
    'MSK_DPAR_INTPNT_CO_TOL_DFEAS', tol, ...
    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP', tol);


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

fval = res.sol.itr.pobjval;


