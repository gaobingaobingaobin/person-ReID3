%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2015-16.  Elyor Kodirov, Vision Group, 
% Queen Mary University of London,
% e.kodirov@qmul.ac.uk.
%
% This is the implementation of following paper:
% Dictionary Learning with Iterative Laplacian Regularisation for
% Unsupervised Person Re-identification, BMVC, 2015.
% version_1: 16/01/2016
% -------------------------------------------------------------------------
% If you use the code, please cite the paper
% -------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ W_full ] = graph_cross_view(X2, X1,k)
% GRAPH_CROSS_VIEW constructs the vross-view graph X2 vs. X1
% X1 is Nxd, X2 is Nxd, N is the number of people, and d is feature
% dimemsion. k is the number of nearest neighbours.

    cosine_dis   = pdist2(X2, X1, 'cosine');    
    W_BA_temp = 1 - cosine_dis;
    sW_BA = zeros(size(cosine_dis));
    for i = 1:316
        [~, idx] = sort(W_BA_temp(i,:), 'descend');
        sW_BA(i,idx(1:k)) = W_BA_temp(i, idx(1:k));
    end
    
    ZERO = zeros(316,316);
    W_full = [ZERO, sW_BA; sW_BA ZERO];        % We don't consider intra-view relationship
    W_full = (W_full + W_full')/2;
    
    W_full(W_full<0.5 & W_full ~= 0) = 0.0001; % Remove noise from the grah
    
    for i1 = 1:632
        W_full(i1,i1) = 0;
    end


end

