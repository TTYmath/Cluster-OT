%% Test marginal OT for Ising model
clear all
close all
addpath('./Tools'); 
addpath('./Solvers/'); 
rng('default'); 
%% parameters 
n = 32; % number of sites
G = spdiags([ones(n,1),zeros(n,1),ones(n,1)],[-1,0,1],n,n);
n1 = size(G,1);
m = n-1;
n = n1;
%% generate first distribution
beta1 = 0.3; %0.6; %0.44; % 0.2 - 0.6 
J1 = -1; 
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
%% generate second distribution
beta2 = 0.3; %0.6; %0.44; % 0.2 - 0.6 
J2 = -1; 
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

%% standard sampling method 
Nlist = [10,ceil(1.5.^(12:23))]; % sample size
err_sample = zeros(1,length(Nlist));
err_mar = zeros(4,length(Nlist));
time_sample = zeros(1,length(Nlist));
time_mar = zeros(4,length(Nlist));
tt = 0;
for N = Nlist
    tt = tt+1;
    X = TT_cross_Ising(n, beta1, h1, J1, 100000); % TT conditional sampling
    X1 = X(1:N,:);
    X2 = X(N+1:2*N,:);
    %% Sinkhorn method
    Cmat = sum(X1.^2,2)+sum(X2'.^2,1)-2*X1*X2';
    M = Cmat;
    a = ones(N,1)/N;
    b = ones(N,1)/N;
    Mscaled = M / median(M(:));
    Tsk = 1e-2;
    maxiter = 1e4;
    tol = 1e-4;
    tic;
    [P, u, v, info] = sinkhorn_log(Mscaled, a, b, Tsk, maxiter, tol);
    LPtime = toc;
    fLP = P(:)'*M(:);
    err_sample(tt) = fLP;
    time_sample(tt) = LPtime;
    fprintf('\n fLP: %6.7e, time: %3.2e',fLP,LPtime);
    %% marginal relaxation method
    for bdw = 1
        K = ceil(n/bdw);
        Ne = cell(K,1);
        for k = 1:K-1
            Ne{k} = (k-1)*bdw+1:k*bdw;
        end
        Ne{K} = k*bdw+1:n;
        Gtype = 2; % 1, empty; 2,path;
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
        [fval,ttime] = Ising_MAR_sample(G,Ne,Gr,X1,X2);
        fprintf('\n Marginal relaxation: bdw: %2d, fval: %6.7e, ttime: %3.2e',bdw,fval,ttime);
        err_mar(bdw,tt) = fval;
        time_mar(bdw,tt) = ttime;
    end
end



fsize = 20;
figure(1)

loglog(Nlist,time_sample,"--ok",'LineWidth', 3, 'MarkerSize', 15)
hold on
loglog(Nlist,time_mar(1,:),"-+k",'LineWidth', 3, 'MarkerSize', 15)


xlabel('sample size','FontSize',fsize); ylabel('running time(s)','FontSize',fsize);
xlim([100 max(Nlist)])
legend({'Vanilla OT','Marginal'}, 'Location','best');






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

