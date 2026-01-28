%% Test cluster moment relaxation for Gaussian distribution
clear all
close all
addpath('./Tools');
addpath('./Solvers/');
addpath(genpath('/Users/tangtianyun/MATLAB/Downloaded/mosek/11.0/'));
Start_up;
rng('default');
%% Parameters:
useSampling = 1;
%% Generate marginal gaussian distribution
n = 20; % number of sites 
G = spdiags(ones(n,2),[-1,1],n,n); % correlative graph: path
m = nnz(G)/2;
[I,J,~] = find(G);
b = randn(2*m,1);
T1 = sparse(I,J,b,n,n);
T1 = (T1+T1')/2;
T1 = T1+spdiags(sum(abs(T1),2)+0.1,0,n,n);
b = randn(2*m,1);
T2 = sparse(I,J,b,n,n);
T2 = (T2+T2')/2;
T2 = T2+spdiags(sum(abs(T2),2)+0.1,0,n,n);
S1 = inv(T1);
mu1 = randn(n,1);
[II,JJ,~] = find(triu(G));
mu2 = randn(n,1);
S2 = inv(T2);
%% Real wasserstain distance
W = norm(mu1-mu2)^2+trace(S1+S2-2*(S1^(0.5)*S2*S1^(0.5))^(0.5));
LPerrlist = [];
LPtimelist = [];
SDPerrlist = [];
SDPtimelist = [];
Nlist = 2.^(5:13); % sample size
for N = Nlist
    fprintf('\n sample size: %2d ',N);
    %% Sinkhorn method
    if useSampling
        X1 = mvnrnd(mu1, S1, N);
        X2 = mvnrnd(mu2, S2, N);
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
        LPerr = abs(fLP-W)/abs(W);
        LPerrlist = [LPerrlist;LPerr];
        LPtimelist = [LPtimelist;LPtime];
        fprintf('\n fLP: %6.7e, LPerr: %3.2e, time: %3.2e',fLP,LPerr,LPtime);
    end
    %% cluster moment relaxation
    err_list = [];
    for k = 5
        G1 = (G+speye(n))^k;
        I = cell(n,1);
        for i = 1:n
            I{i} = i;
        end
        [fSDP,~,SDPtime] = OT_Moment_relax_sparse(G1,I,X1,X2,1);
        SDPerr = abs(fSDP-W)/abs(W);
        SDPerrlist = [SDPerrlist;SDPerr];
        SDPtimelist = [SDPtimelist;SDPtime];
        fprintf('\n fSDP: %6.7e, error : %6.7e, time: %3.2e',fSDP,SDPerr,SDPtime);
    end
end


fsize = 20;
figure(1)
h4 = loglog(Nlist,LPtimelist,"--ok",'LineWidth', 3, 'MarkerSize', 15);
hold on
h5 = loglog(Nlist,SDPtimelist,"-+k",'LineWidth', 3, 'MarkerSize', 15);
NN = 1000:1000:5000;
%h6 = loglog(NN,(NN/800).^1.5,'Color','b','LineStyle','--','LineWidth',1.5);
xlim([min(Nlist),max(Nlist)]);
xlabel('sample size','FontSize',fsize); ylabel('running time(s)','FontSize',fsize);
title('dimension = 20','FontSize',fsize, 'FontWeight', 'normal');
legend([h4,h5],{'Vanilla OT','Cluster Moment'}, 'FontSize', 10, 'Location','best');


