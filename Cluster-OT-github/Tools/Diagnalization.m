function [qm,Qm,Qf,q] = Diagnalization(G,C,c,T)
n = size(G,1);
r = length(c{1});
if r^(2*n) > 1e8
    error('\n too large') 
end
[I,J,~] = find(triu(G));
m = length(I);
n = size(G,1);
q = zeros(r^n,1);
for i = 1:r^n
    % construct index
    s = dec2base(i-1,r);
    s = s-'0';
    if length(s)<n
        s = [zeros(1,n-length(s)),s];
    end
    s = s+1;
    for k = 1:n
        q(i) = q(i)+c{k}(s(k));
    end
    for k = 1:m
        ii = I(k);
        jj = J(k);
        q(i) = q(i)+C{k}(s(ii),s(jj));
    end
end

q = exp(-q/T);
q = q/sum(q);

qm = zeros(r,n);
Qm = zeros(r,r,m);
Qf = cell(n,n);
for j = 1:n
    for i = 1:j
        Qf{i,j} = zeros(r,r);
    end
end
% Qm = cell(m,1);
% for i = 1:m
%     Qm{i} = zeros(r,r);
% end

for i = 1:r^n
    % construct index
    s = dec2base(i-1,r);
    s = s-'0';
    if length(s)<n
        s = [zeros(1,n-length(s)),s];
    end
    s = s+1;
    for k = 1:n
        qm(s(k),k) = qm(s(k),k)+q(i);
    end
    for k = 1:m
        ii = I(k);
        jj = J(k);
        Qm(s(ii),s(jj),k) = Qm(s(ii),s(jj),k)+q(i);
    end
    for jj = 1:n
        for ii = 1:jj
            Qf{ii,jj}(s(ii),s(jj)) = Qf{ii,jj}(s(ii),s(jj))+q(i);
        end
    end
end






