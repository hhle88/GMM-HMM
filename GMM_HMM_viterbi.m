function [chain_opt, fopt] = GMM_HMM_viterbi(mean, var, weight, aij, obs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testing for isolated digital models (0, 1,..., 9),

if nargin==0
    load 'models\HMM_30.mat'; % load a GMM-HMM
    digit = 1; % 0 <= digit <= 9
    mean = HMM.mean(:,:,:,digit);
    var = HMM.var(:,:,:,digit);
    weight = HMM.weight(:,:,digit);
    aij = HMM.Aij(:,:,digit);
    [DIM, ~, ~, ~] = size(mean);
    obs = rand(DIM,30);
end
aij = cat(3,aij,aij*aij,aij*aij*aij); % in case frame loss
[DIM, T] = size(obs); % T: length of observations or number of observation frames
[~, num_of_mix, num_of_state, ~] = size(mean); % num_of_state: NOT including START and END states (nodes) in HMM
num_of_state = num_of_state + 2; % number of states, including START and END states (nodes) in HMM
mean_temp = NaN(DIM, num_of_mix, num_of_state);
var_temp = NaN(DIM, num_of_mix, num_of_state);
for i = 1:num_of_state-2
    mean_temp(:,:,i+1) = mean(:,:,i);
    var_temp(:,:,i+1) = var(:,:,i);
end
mean = mean_temp; var = var_temp; % insert value NaN for the state START and END
weight = [NaN(1,num_of_mix); weight; NaN(1,num_of_mix)];
aij(end,end) = 1;
timing = 1:T+1;
num_of_state = size(mean, 3);
fjt = -Inf(num_of_state, T);
s_chain = cell(num_of_state, T);

%%%%%% at t = 1
dt = timing(1);
for j=2:num_of_state-1 % 2->14
    fjt(j,1) = log(aij(1,j,dt)) + log_mul_Gau(mean(:,:,j),var(:,:,j),weight(j,:), obs(:,1));
    if fjt(j,1) > -Inf
        s_chain{j,1} = [1 j];
    end
end

for t=2:T
    dt = timing(t)-timing(t-1); % in case frame loss and dt = 2, 3, 4,...
    for j=2:num_of_state-1 %(2->14)
        f_max = -Inf;
        i_max = -1;
        f = -Inf;
        for i=2:j
            if(fjt(i,t-1) > -Inf)
                f = fjt(i,t-1) + log(aij(i,j,dt)) + log_mul_Gau(mean(:,:,j),var(:,:,j),weight(j,:), obs(:,t));
            end
            if f > f_max % finding the f max
                f_max = f;
                i_max = i; % index
            end
        end
        if i_max ~= -1
            s_chain{j,t} = [s_chain{i_max,t-1} j];
            fjt(j,t) = f_max;
        end
    end
end
%%%%%% at t = end
dt = timing(end) - timing(end - 1); % in case frame loss and dt = 2, 3, 4,...
fopt = -Inf;
iopt = -1;
for i=2:num_of_state-1
    f = fjt(i, T) + log(aij(i, num_of_state, dt));
    if f > fopt
        fopt = f;
        iopt = i;
    end
end
%%%%%% optimal result
% fjt
% fopt
if iopt ~=-1
    chain_opt = [s_chain{iopt,t} num_of_state];
end
end
%% this function is for the case of multiple Gaussian HMM
function log_b = log_mul_Gau (mean_i, var_i, c_j, o_i)
[dim, num_of_mix] = size(mean_i);
log_N = -Inf(1,num_of_mix);
log_c = -Inf(1,num_of_mix);
for m = 1:num_of_mix
    log_N(m) = -1/2*(dim*log(2*pi) + sum(log(var_i(:,m))) + sum((o_i - mean_i(:,m)).*(o_i - mean_i(:,m))./var_i(:,m)));
    log_c(m) = log(c_j(m));
end
y = -Inf(1,num_of_mix);
ymax = -Inf;
for m = 1:num_of_mix
    y(m) = log_N(m) + log_c(m);
    if y(m) > ymax
        ymax = y(m);
    end
end
if ymax == Inf
    log_b = Inf;
else
    sum_exp = 0;
    for m = 1:num_of_mix
        if ymax == -Inf && y(m) == -Inf
            sum_exp = sum_exp + 1;
        else
            sum_exp = sum_exp + exp(y(m) - ymax);
        end
    end
    log_b = ymax + log(sum_exp);
end
end

%% this function is for case of single Gaussian HMM
% function log_b = logGaussian (mean_i, var_i, o_i)
% dim = length(var_i);
% log_b = -1/2*(dim*log(2*pi) + sum(log(var_i)) + sum((o_i - mean_i).*(o_i - mean_i)./var_i));
% end
