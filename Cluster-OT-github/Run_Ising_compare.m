%% Test marginal OT for 1-D Ising model
clear all
close all
addpath('./Tools');
addpath('./Solvers/');
rng('default');
Nlist = 2.^(2:7);
err_sample = zeros(1,length(Nlist));
err_mar = zeros(1,length(Nlist));
time_sample = zeros(1,length(Nlist));
time_mar = zeros(1,length(Nlist));
tt = 0;
for n = Nlist
    Atype = 4; % 1, 2D grid; 2, 3D grid; 3, random; 4, path
    if Atype == 1 % 2D
        G = gen2Dgrid(n);
        n1 = size(G,1);
        m = nnz(G)/2;
    elseif Atype == 2 % 3D
        G = gen3Dlattice(n);
        n1 = size(G,1);
        m = nnz(G)/2;
    elseif Atype == 3
        G = random_graph_m(n,3*n);
        n1 = size(G,1);
        m = nnz(G)/2;
    else
        G = spdiags([ones(n,1),zeros(n,1),ones(n,1)],[-1,0,1],n,n);
        n1 = size(G,1);
        m = n-1;
    end
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
    N = 10000; 
    tt = tt+1; 
    X0 = TT_cross_Ising(n, beta1, h1, J1, 50000); % TT cross sampling
    X = X0(end-2*N+1:end,:); 
    X1 = X(1:N,:); 
    X2 = X(N+1:2*N,:); 
    %% Sinkhorn method
    if 2^n >= N
        M = sum(X1.^2,2)+sum(X2'.^2,1)-2*X1*X2';
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
    else
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
        q1s = zeros(2^n,1);
        for i = 1:2^n
            % construct index
            s = dec2base(i-1,2);
            s = s-'0';
            if length(s)<n
                s = [zeros(1,n-length(s)),s];
            end
            s = 2*s-1;
            q1s(i) = sum(sum(abs(X1-s),2)<0.5)/N;
        end
        q2s= zeros(2^n,1);
        for i = 1:2^n
            % construct index
            s = dec2base(i-1,2);
            s = s-'0';
            if length(s)<n
                s = [zeros(1,n-length(s)),s];
            end
            s = 2*s-1;
            q2s(i) = sum(sum(abs(X2-s),2)<0.5)/N;
        end
        M = Cmat;
        a = q1s;
        b = q2s;
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
    end
    %% marginal relaxation method
    bdw = 1; % cluster size
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
    X = X0(end-2*N+1:end,:);
    X1 = X(1:N,:);
    X2 = X(N+1:2*N,:);
    [fval,ttime] = Ising_MAR_sample(G,Ne,Gr,X1,X2);
    fprintf('\n Marginal relaxation: bdw: %2d, fval: %6.7e, ttime: %3.2e',bdw,fval,ttime);
    err_mar(bdw,tt) = fval;
    time_mar(bdw,tt) = ttime;
end



fsize = 20;
figure(1)
subplot(1,2,1)
loglog(Nlist,err_sample,"--ok",'LineWidth', 3, 'MarkerSize', 15)
hold on
loglog(Nlist,err_mar(1,:),"-+k",'LineWidth', 3, 'MarkerSize', 15)
NN = 40:100;
loglog(NN,NN/25,"--b",'LineWidth', 1.5, 'MarkerSize', 15)
xlabel('dimension','FontSize',fsize); ylabel('OT cost','FontSize',fsize);
title('sample size = 10000','FontSize',fsize, 'FontWeight', 'normal');
xlim([min(Nlist) max(Nlist)])
legend({'Vanilla OT','Marginal','O(d)'}, 'Location','best','FontSize', 10);


subplot(1,2,2)

loglog(Nlist,time_sample,"--ok",'LineWidth', 3, 'MarkerSize', 15)
hold on
loglog(Nlist,time_mar,"-+k",'LineWidth', 3, 'MarkerSize', 15)
NN = 40:100;
loglog(NN,NN/4000,"--b",'LineWidth', 1.5, 'MarkerSize', 15)
xlabel('dimension','FontSize',fsize); ylabel('running time(s)','FontSize',fsize);
title('sample size = 10000','FontSize',fsize, 'FontWeight', 'normal');
xlim([min(Nlist) max(Nlist)])
legend({'Vanilla OT','Marginal','O(d)'}, 'Location','best','FontSize', 10);






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

