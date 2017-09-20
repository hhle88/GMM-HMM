function HMM = EM_HMMtraining_multiGaussian(training_file_list, DIM, num_of_model, num_of_state)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function deals with GMM-HMM training problem. The programe will run as follow: initialization HMM,
% re-estimating HMM, mixture spliting, re-estimating HMM, mixture spliting, so on. Depend on number of 
% mixture we want to split and how many iterations we want to re-estimating, user may modify procedure by himself.

if nargin==0
    DIM = 39;
    num_of_model = 10;
    num_of_state = 13; % 13 nodes, and not including START and END node
    training_file_list = 'trainingfile_list.mat';
end
!rd models /s /q
!md models
% generate initial HMM or global means, vars
HMM = EM_initialization_model(training_file_list, DIM, num_of_state, num_of_model);

load (training_file_list, 'trainingfile');
% log_likelihood_iter = zeros(1, num_of_iteration);
% likelihood_iter = zeros(1, num_of_iteration);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% re-estimating
for iter = 1:5
    HMM = EM_hmm_multi_gau(HMM, trainingfile);
    save_HMM_to_a_file(HMM, iter);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% spliting
HMM = spliting_HMM(HMM); save_HMM_to_a_file(HMM, iter+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% re-estimating
for iter = 7:10
    HMM = EM_hmm_multi_gau(HMM, trainingfile);
    save_HMM_to_a_file(HMM, iter);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% spliting
HMM = spliting_HMM(HMM); save_HMM_to_a_file(HMM, iter+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% re-estimating
for iter = 12:20
    HMM = EM_hmm_multi_gau(HMM, trainingfile);
    save_HMM_to_a_file(HMM, iter);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% spliting
HMM = spliting_HMM(HMM); save_HMM_to_a_file(HMM, iter+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% re-estimating
for iter = 22:30
    HMM = EM_hmm_multi_gau(HMM, trainingfile);
    save_HMM_to_a_file(HMM, iter);
end

% % % % % log_likelihood_iter
% % % % % likelihood_iter
% figure();
% plot(log_likelihood_iter,'-*');
% xlabel('iterations'); ylabel('log likelihood');
% title(['number of states: ', num2str(num_of_state)]);

%% print HMM to a file
save_file_name = 'HMM_GMM.txt';
file_list_of_model = 'list_of_models.txt';
printf_HMM_to_file(HMM,save_file_name,file_list_of_model);
end

%%
function HMM = EM_initialization_model(training_file_list, DIM, num_of_state, num_of_model)
HMM.mean = zeros(DIM, 1, num_of_state, num_of_model);
HMM.var  =  zeros(DIM, 1, num_of_state, num_of_model);
HMM.Aij  = zeros(num_of_state+2, num_of_state+2, num_of_model);
HMM.weight = ones(num_of_state,1,num_of_model);

sum_of_features = zeros(DIM,1);
sum_of_features_square = zeros(DIM, 1);
num_of_feature = 0;

load (training_file_list, 'trainingfile');
num_of_uter = size(trainingfile,1);

parfor u = 1:num_of_uter
    filename = trainingfile{u,2};
    mfcfile = fopen(filename, 'r', 'b' );
    if mfcfile ~= -1
        nSamples = fread(mfcfile, 1, 'int32');
        sampPeriod = fread(mfcfile, 1, 'int32')*1E-7;
        sampSize =fread(mfcfile, 1, 'int16');
        dim = 0.25*sampSize; % dim = 39
        parmKind = fread(mfcfile, 1, 'int16');
        
        features = fread(mfcfile, [dim, nSamples], 'float');
        
        sum_of_features = sum_of_features + sum(features, 2); % for calculating mean
        sum_of_features_square = sum_of_features_square + sum(features.^2, 2); % for calculating variance
        num_of_feature = num_of_feature + size(features,2); % number of elements (feature vectors) in state m of model k
        
        fclose(mfcfile);
    end    
end
% calculate value of means, variances, aijs
HMM = calculate_inital_mean_var_aij(HMM, num_of_state, num_of_model, sum_of_features, sum_of_features_square, num_of_feature);
save_HMM_to_a_file(HMM, 0);
end

%%
function HMM = calculate_inital_mean_var_aij(HMM, num_of_state, num_of_model, sum_of_features, sum_of_features_square, num_of_feature)

for k = 1:num_of_model
    for m = 1:num_of_state
        HMM.mean(:,1,m,k) = sum_of_features/num_of_feature;
        HMM.var(:,1,m,k) = sum_of_features_square/num_of_feature - HMM.mean(:,1,m,k).*HMM.mean(:,1,m,k);
    end
    for i = 2:num_of_state+1
        HMM.Aij(i,i+1,k) = 0.4;
        HMM.Aij(i,i,k) = 1-HMM.Aij(i,i+1,k);        
    end
    HMM.Aij(1,2,k) = 1;
end
end
