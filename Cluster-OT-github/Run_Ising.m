%% Test marginal OT for 1-D Ising model
clear all
close all
addpath('./Tools');
addpath(genpath('./Solvers/'));
rng('default');
%% parameters 
n = 12; % number of sites, it is "d" in the paper
G = spdiags([ones(n,1),zeros(n,1),ones(n,1)],[-1,0,1],n,n); % adjacency of path graph
n1 = size(G,1);
m = n-1;
n = n1;
%% generate source distribution
beta1 = 0.6; %0.6; %0.44; % 0.2 - 0.6
J1 = 1;
h1 = 0.2;
T1 = 1/beta1;
C1 = cell(m,1);
c1 = cell(n,1);
for i = 1:n
    c1{i} = -h1*[1;-1];
end
for i = 1:m
    C1{i} = -J1*[1,-1;-1,1];
end
[qD1,QD1,Qf1,q1] = Diagnalization(G,C1,c1,T1);
%% generate terget distribution
beta2 = 0.2; %0.6; %0.44; % 0.2 - 0.6
J2 = 1;
h2 = 0.2;
T2 = 1/beta2;
C2 = cell(m,1);
c2 = cell(n,1);
for i = 1:n
    c2{i} = -h2*[1;-1];
end
for i = 1:m
    C2{i} = -J2*[1,-1;-1,1];
end
[qD2,QD2,Qf2,q2] = Diagnalization(G,C2,c2,T2);


%% exact solution for solving an LP problem

% cost function
Cmat = zeros(2^n,2^n);
for i = 1:2^n
    % construct index
    s1 = dec2base(i-1,2);
    s1 = s1-'0';
    if length(s1)<n
        s1 = [zeros(1,n-length(s1)),s1];
    end
    s1 = 1-2*s1;
    for j = 1:2^n
        s2 = dec2base(j-1,2);
        s2 = s2-'0';
        if length(s2)<n
            s2 = [zeros(1,n-length(s2)),s2];
        end
        s2 = 1-2*s2;
        Cmat(i,j) = norm(s1-s2)^2;
    end
end

compute_exact = 1;
if compute_exact
    N1 = 2^n;
    if N1>1e4
        error('\n too large');
    end
    A = [kron(speye(N1),ones(1,N1));kron(ones(1,N1),speye(N1))];
    b = [q2;q1];
    cc = Cmat(:);
    prob.c = cc;
    prob.a = sparse(A);
    prob.blc = b;
    prob.buc = b;
    prob.blx = zeros(N1^2,1);
    prob.bux = inf*ones(N1^2,1);
    param = [];
    param.MSK_IPAR_LOG            = 2;      % print per‐iteration info
    param.MSK_CALLBACK_FUNC       = @mycallback;
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
    ttimeo = toc;
    optvalo = res.sol.bas.pobjval;
    fprintf('\n real W2 distance: %6.7e, time = %3.2e',res.sol.bas.pobjval,ttimeo);
end

%% marginal relaxation method
fvalk = [];
bdk = [];
ttimek = [];
for bdw = 1:4 % cluster size
    K = ceil(n/bdw);
    Ne = cell(K,1);
    for k = 1:K-1
        Ne{k} = (k-1)*bdw+1:k*bdw;
    end
    Ne{K} = k*bdw+1:n;
    Gtype = 2; % correlative graph: 1, empty; 2,path;
    if Gtype == 1
        Gr = zeros(K,K);
    elseif Gtype == 2
        Gr = zeros(K,K);
        for i = 1:K-1
            Gr(i,i+1) = 1;
            Gr(i+1,i) = 1;
        end
    else
        error('\n Gtype error');
    end
    [fval,ttime] = Ising_MAR(G,Ne,Gr,C1,c1,T1,C2,c2,T2);
    fvalk = [fvalk,fval];
    ttimek = [ttimek,ttime];
    bdk = [bdk,bdw];
    fprintf('\n Marginal relaxation: bdw: %2d, fval: %6.7e, ttime: %3.2e',bdw,fval,ttime);
end

fprintf('\n Ising model:\n beta1,J1,h1 = %3.2e,%3.2e,%3.2e,\n beta2,J2,h2 = %3.2e,%3.2e,%3.2e',beta1,J1,h1,beta2,J2,h2);



fprintf('\n real W2 distance: %6.7e, time = %3.2e',optvalo,ttimeo);
for k = 1:length(fvalk)
    fprintf('\n Marginal relaxation: bdw: %2d, fval: %6.7e, ttime: %3.2e',bdk(k),fvalk(k),ttimek(k));
end



function stop = mycallback(task, where)
% Called by MOSEK at various points; we want IPM iterations
persistent objs
stop = false;  % returning true would halt the solve

% 15 == MSK_CALLBACK.AFTER_INTPNT_ITERATION
if where == 15
    % grab the current primal objective
    [rcode, pobj] = mosekopt('getdouinf MSK_DINF_INTPNT_PRIMAL_OBJ');
    if rcode == 0
        objs(end+1) = pobj;
        % live‐update plot
        figure(1);
        plot(objs, '-o','LineWidth',1.5);
        xlabel('IPM Iteration');
        ylabel('Primal Objective');
        title('MOSEK IPM Progress');
        grid on;
        drawnow;
    end
end
end

