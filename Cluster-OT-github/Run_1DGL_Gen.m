%% Test cluster moment relaxation generative modeling of 1D Ginzburg Landau model
clear all 
close all
addpath('./Tools/'); 
addpath(genpath('./Solvers/')); 
rng('default');  
%% Parameters:   
d = 10; % number of sites 
N = 100000; % test sample size 
N1 = 10000; % train sample size 

%% Generate samples
% target distribution: 1-D Ginzburg Landau
beta = 1/8; 
lambda = 0.06; 
L = 2.5; 
dx = 0.02; 
h = 1/(d+1); 
% TT conditional sampling of 1D Ginzburg Landau distribution
sampley = TT_sample_1GLD(d,h, beta, lambda, L, dx, N+N1); 
train2 = sampley(1:N1,:); % training set y
test2 = sampley(N1+1:end,:); % testing set y

% source distribution: Gaussian
mu = mean(train2,1);
Sigma = cov(train2,1);
train1 = mvnrnd(mu, Sigma, N1);
test1 = mvnrnd(mu, Sigma, N);

%% Cluster moment relaxation

bdw = 1; % cluster size 
hh = 1; % correlative radius
deg = 10; % relaxation degree 

K = ceil(d/bdw); % cluster number
if K > 1
    Ne = cell(K,1);
    for k = 1:K-1
        Ne{k} = (k-1)*bdw+1:k*bdw;
    end
    Ne{K} = k*bdw+1:d;
else
    Ne = {1:d};
end

G = spdiags([ones(K,1),zeros(K,1),ones(K,1)],[-1,0,1],K,K); % correlation graph
G = G+speye(K);
G = G^hh;
G = G-diag(diag(G))>0;


% Run OT cluster moment relaxation to get OT cost and transport map
[fval,Tmap] = OT_Moment_relax_sparse(G,Ne,train1,train2,deg);


% Apply transport map to testing set
test121 = zeros(N,d);
for j = 1:N
    test121(j,:) = Tmap(test1(j,:));
end
test12 = min(max(-L,test121),L);

%% plot of 2 marginal
figure(1)
clf
mar_dim = [1,2];
fsize = 20;

%% ---------- First Row ----------
subplot(1,3,1);
histogram2(test1(:,mar_dim(1)), test1(:,mar_dim(2)), 50, ...
    'DisplayStyle','tile','ShowEmptyBins','off', ...
    'XBinLimits',[-L L],'YBinLimits',[-L L]);
xlabel('x_1','FontSize',fsize);
ylabel('x_{20}','FontSize',fsize);
title('Source','FontSize',fsize, 'FontWeight', 'normal');

subplot(1,3,2);
histogram2(test2(:,mar_dim(1)), test2(:,mar_dim(2)), 50, ...
    'DisplayStyle','tile','ShowEmptyBins','off', ...
    'XBinLimits',[-L L],'YBinLimits',[-L L]);
xlabel('y_1','FontSize',fsize);
ylabel('y_{20}','FontSize',fsize);
title('Target','FontSize',fsize, 'FontWeight', 'normal');

subplot(1,3,3);
histogram2(test12(:,mar_dim(1)), test12(:,mar_dim(2)), 50, ...
    'DisplayStyle','tile','ShowEmptyBins','off', ...
    'XBinLimits',[-L L],'YBinLimits',[-L L]);
xlabel('y_1','FontSize',fsize);
ylabel('y_{20}','FontSize',fsize);
title('Cluster Moment n=1','FontSize',fsize, 'FontWeight', 'normal');

% ---- Add one colorbar for the entire first row ----
cb1 = colorbar;
cb1.Position = [0.92 0.60 0.015 0.33];   % [x y width height]
