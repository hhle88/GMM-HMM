function HMM_new = spliting_HMM(HMM)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is for mixture incrementing or mixture splitting
 
if nargin == 0
    load('models\HMM_1.mat','HMM');
end
[DIM, num_of_recent_mix, num_of_state, num_of_model] = size(HMM.mean);
mean = zeros(DIM, num_of_recent_mix+1, num_of_state, num_of_model);
var = zeros(DIM, num_of_recent_mix+1, num_of_state, num_of_model);
weight = zeros(num_of_state, num_of_recent_mix+1, num_of_model);
HMM_new.mean = mean;
HMM_new.var = var;
HMM_new.weight = weight;
HMM_new.Aij = HMM.Aij;
for k = 1:num_of_model
    for i = 1:num_of_state
        [~, id_max] = max(HMM.weight(i,:,k));
        HMM_new = spliting_for_each_state(HMM_new, HMM, id_max, i, k);
    end
end
end

%%
function HMM_new = spliting_for_each_state(HMM_new, HMM, id_max, i, k)
rate = 0.2; % according to HTK book (page 157)
[~, num_of_recent_mix, ~, ~] = size(HMM.mean);
%%
HMM_new.mean(:,1:num_of_recent_mix,i,k) = HMM.mean(:,1:num_of_recent_mix,i,k);
HMM_new.mean(:,num_of_recent_mix+1,i,k) = HMM.mean(:,id_max,i,k) - rate*sqrt(HMM.var(:,id_max,i,k));
HMM_new.mean(:,id_max,i,k) = HMM.mean(:,id_max,i,k) + rate*sqrt(HMM.var(:,id_max,i,k));
%%
HMM_new.var(:,1:num_of_recent_mix,i,k) = HMM.var(:,1:num_of_recent_mix,i,k);
HMM_new.var(:,num_of_recent_mix+1,i,k) = HMM.var(:,id_max,i,k);
%%
HMM_new.weight(i,1:num_of_recent_mix,k) = HMM.weight(i,1:num_of_recent_mix,k);
HMM_new.weight(i,num_of_recent_mix+1,k) = HMM.weight(i,id_max,k)/2;
HMM_new.weight(i,id_max,k) = HMM.weight(i,id_max,k)/2;
end