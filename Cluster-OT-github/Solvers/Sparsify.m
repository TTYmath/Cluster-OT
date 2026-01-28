function [blk,At,C,b,CSet] = Sparsify(blk,At,C,b)
[A, b, c, K] = sdpt3_to_sedumi(blk, At, C, b);
clear J
J.f = length(b);
parCoLO.SDPsolver = [];


if exist('sparseCoLO','file') ~= 2
    error(sprintf([ ...
        'SparseCoLO not found on MATLAB path.\n\n', ...
        'This function requires SparseCoLO (GPL v2).\n', ...
        'Please download it from:\n', ...
        '  http://www.opt.c.titech.ac.jp/kojima/SparseCoLO/SparseCoLO.htm\n\n', ...
        'and add it to your MATLAB path before calling Sparsify().' ...
        ]));
end

[x,y,infoCoLO,cliqueDomain,cliqueRange,LOP] = sparseCoLO(A,b,c,K,J,parCoLO);
A = LOP.A;
b = LOP.b;
c = LOP.c;
K = LOP.K;
[blk, At, C, b] = sedumi_to_sdpt3(A, b, c, K);
CSet = cliqueDomain{1}.Set;






