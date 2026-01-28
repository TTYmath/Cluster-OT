%% Test Cluster moment relaxation for Gaussian distribution
clear all 
close all
addpath('./Tools'); 
addpath(genpath('./Solvers/')); 
rng('default');  
%% Parameters:        
useSampling = 1; 
%% Generate marginal gaussian distribution   
n = 100; % number of sites, it is "d" in the paper
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
%% Cluster Moment relaxation with exact moment
err_list = [];
for k = 1:10
    G1 = (G+speye(n))^k;
    I = cell(n,1);
    for i = 1:n
        I{i} = i;
    end    
    [fval,ttime] = GGM_SOS(G1,S1,S2,mu1,mu2);
    err = abs(fval-W)/abs(W);
    err_list = [err_list,err];
    fprintf('\n h: %2d, fval: %6.7e, error: %3.2e, time: %3.2e',k,fval,err,ttime);
end

figure(1)
fsize = 35;
figure(1)
semilogy(1:10, err_list, '-+k', 'LineWidth', 3, 'MarkerSize', 15);
hold on
xlabel('correlation radius (h)','FontSize',fsize); ylabel('relative error','FontSize',fsize);
title('dimension = 100','FontSize',fsize, 'FontWeight', 'normal');


