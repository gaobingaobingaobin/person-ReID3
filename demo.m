%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2017.  Bin Gao 
% Key Laboratory of Underwater Acoustic Signal, Processing of Ministry of Education
% Southeast University, Nanjing
%
% feimaxiao123@gmail.com
%

% Our code of GRDL2 is inspired and modified from the matlab code of following paper:
%
%
% Elyor Kodirov, Tao Xiang and Shaogang Gong. 
% Dictionary Learning with Iterative Laplacian Regularisation for Unsupervised Person Re-identification. 
% In Xianghua Xie, Mark W. Jones, and Gary K. L. Tam, editors, 
% Proceedings of the British Machine Vision Conference (BMVC), 
% pages 44.1-44.12. BMVA Press, September 2015.
%
% version_1: 9/04/2017
% -------------------------------------------------------------------------
% If you use the code, please cite our el paper:
% -------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Bin Gao, Mingyong Zeng, et al., 'Person Re-identification by unsupervised Laplacian graph learning based
% on  projective weight', Electronic letters, submmited, 2017


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By running this demo, you are supposed to get about rank 1 with 32.27%
% load the data
clc
clear all
rand('twister', 5489); warning('off', 'all');

% load the data
load('VIPeR_data_trial_1.mat');
acc           = [];
for k_nn=7
     
% k_nn        = 4;  

%%
disp('Start, Unsupervised ...');
for out_iter = 1:2

    disp(strcat('Iter: ',num2str(out_iter)));
    if out_iter == 1
        X1 = Cam_A_tr;
        X2 = Cam_B_tr;
    else
        % Use the learned coefficients this time
        X1 = train_a_after_D';
        X2 = train_b_after_D';
    end
    
    % Graph construciton
    [W_full] = graph_cross_view( X2,X1,k_nn);
    disp('1. Graph construction (W) is finished.');
    
    %% Learning dictionary
    X_tr   = [Cam_A_tr; Cam_B_tr]';
    nBasis = power(2, 8); % number of dictionary atoms
    alpha  = 1;           % graph constraint
    beta   = .0001;       % sparsity constraint
    nIters = 50;          % number iteration
    [D, ~, ~] = GraphSC_cor(X_tr, W_full, nBasis, alpha, beta, nIters);      
    disp('2. Learning dictionary (D) is finished.');
    
    %% Train data coefficients
    lambda1 = .04; % This parametr must be tuned 
    P              = pinv( D'*D + lambda1*eye(size(D,2)))*D';
    train_a_after_D = P* Cam_A_tr';
    train_b_after_D = P* Cam_B_tr';
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Testing: generating CMC curve.
disp('3. Results ... ');
% L2 regularisation, one can use Lasso here
lambda1        = .04;        % one can use cross-validation to tune this parameter.
P              = pinv( D'*D + lambda1*eye(size(D,2)))*D';
test_a_after_D = P* Cam_A_te;
test_b_after_D = P* Cam_B_te;

num_p = 316;
maxNumTemplate = 1;
num_gallery    = 316;
num_test       = num_p;

idxProbe   = 1:316;    
idxGallery = 1:316;    

scores_after = pdist2(test_a_after_D', test_b_after_D','cosine'); % Note that we are doing camera B vs. Camera A.
cmc_nn = zeros(num_gallery,3);
cmc_nn(:,1) = 1:num_gallery;
cmcCurrent = zeros(num_gallery,3);
cmcCurrent(:,1) = 1:num_gallery;

for k=1:num_test
    finalScore = scores_after(:,k);
    [sortScore sortIndex] = sort(finalScore);
    [cmc_nn cmcCurrent] = evaluateCMC_demo(idxProbe(k),idxGallery(sortIndex),cmc_nn,cmcCurrent);
end
ranks = cmc_nn(:,2)./cmc_nn(:,3)*100;

% Display results
disp(strcat('Knn number: ',num2str(k_nn), '.'));
disp(strcat('Rank#1: ',num2str(ranks(1)), '.'));
disp(strcat('Rank#5: ',num2str(ranks(5)), '.'));
disp(strcat('Rank#10: ',num2str(ranks(10)), '.'));
disp(strcat('Rank#20: ',num2str(ranks(20)), '.'));
plot(ranks(1:50));
end



