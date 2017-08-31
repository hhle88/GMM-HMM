function [mean_numerator, var_numerator, mean_var_denominator, wei_numerator, wei_denominator, aij_numerator, log_likelihood, likelihood] = forward_backward_hmm_mulgau_log_math(mean, var, aij, weight, obs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin==0
    load mean;
    load var;
    load weight;
    load aij;
    load obs;
end
[dim, T] = size(obs); % T: length of observations or number of observation frames
[~, num_of_mix, num_of_state, ~] = size(mean); % num_of_state: NOT including START and END states (nodes) in HMM
num_of_state = num_of_state + 2; % number of states, including START and END states (nodes) in HMM
mean_temp = NaN(dim,num_of_mix,num_of_state);
var_temp = NaN(dim,num_of_mix,num_of_state);
for i = 1:num_of_state-2
    mean_temp(:,:,i+1) = mean(:,:,i);
    var_temp(:,:,i+1) = var(:,:,i);
end
mean = mean_temp; var = var_temp; % insert value NaN for the state START and END
weight = [NaN(1,num_of_mix); weight; NaN(1,num_of_mix)];
aij(end,end) = 1;
log_alpha = -Inf(num_of_state, T+1); % initialization 
log_beta = -Inf(num_of_state, T+1);  % initialization 
%% calculate log_N_jkt(j,k,t)
log_N_jkt = -Inf(num_of_state,num_of_mix,T); % log single Gaussian for each mixture k in each state j at time step t
for j = 1:num_of_state
    for k = 1:num_of_mix
        for t = 1:T
            log_N_jkt(j,k,t) = -1/2*(dim*log(2*pi) + sum(log(var(:,k,j))) + sum((obs(:,t) - mean(:,k,j)).*(obs(:,t) - mean(:,k,j))./var(:,k,j)));
        end
    end
end

%% calculate alpha ( log(alpha), in fact), !!! notice alpha(1:state_NO, 1:T+1)
for i = 2:num_of_state-1 % from state 1 (START) to END at time step 1
    log_alpha(i,1) = log(aij(1,i)) + log_mul_Gau(mean(:,:,i),var(:,:,i),obs(:,1), weight(i,:)); % log(alpha)
end

for t = 2:T % calculate alpha
    for j = 2:num_of_state-1    
        log_alpha(j,t) = log_sum_alpha(log_alpha(2:num_of_state-1,t-1),aij(2:num_of_state-1,j)) + log_mul_Gau(mean(:,:,j),var(:,:,j),obs(:,t), weight(j,:));
    end
end

log_alpha(num_of_state,T+1) = log_sum_alpha(log_alpha(2:num_of_state-1,T),aij(2:num_of_state-1,num_of_state)); % this value is  P(o1, o2,... , oT | lamda) also

%% calculate beta   (end,end)= (state_NO,T+1), !!! notice beta(1:state_NO, 0:T)
% beta(end,end) = 0; % not used
log_beta(:,T) = log(aij(:,num_of_state));
for t = T-1:-1:1 % calculate beta
    for i = 2:num_of_state-1
        log_beta(i,t) = log_sum_beta(aij(i,2:num_of_state-1),mean(:,:,2:num_of_state-1),var(:,:,2:num_of_state-1),obs(:,t+1),weight(2:num_of_state-1,:),log_beta(2:num_of_state-1,t+1));
    end
end
log_beta(num_of_state,1) = log_sum_beta(aij(1,2:num_of_state-1),mean(:,:,2:num_of_state-1),var(:,:,2:num_of_state-1),obs(:,1),weight(2:num_of_state-1,:),log_beta(2:num_of_state-1,1));

%% calculate Xi(1:num_of_state, 1:num_of_state, 0:T)
log_Xi = -Inf(num_of_state,num_of_state,T);
for t = 1:T-1
    for j = 2:num_of_state-1
        for i = 2:num_of_state-1
            log_Xi(i,j,t) = log_alpha(i,t) + log(aij(i,j)) + log_mul_Gau(mean(:,:,j),var(:,:,j),obs(:,t+1), weight(j,:)) + log_beta(j,t+1) - log_alpha(num_of_state,T+1);
        end
    end
end
%%% when t=T;
for i = 1:num_of_state
    log_Xi(i,num_of_state,T) = log_alpha(i,T) + log(aij(i,num_of_state)) - log_alpha(num_of_state, T+1);
end
%%% when t=0 -> not used
% for j = 1:num_of_state
%     log_Xi(1,j,0) = log_alpha(1,j) + log_beta(j,1) - log_alpha(num_of_state, T+1);
% end

%% calculate log(sum of alpha x beta)
logsumalphabeta = -Inf(1,T);
for t = 1:T
    logsumalphabeta(t) = log_sum_alpha_beta(log_alpha(:,t),log_beta(:,t));
end
%% calculate gamma
log_gamma = -inf(num_of_state,num_of_mix,T);
for t = 1:T
    for j = 2:num_of_state-1
        for k = 1:num_of_mix
            log_gamma(j,k,t) = log_alpha(j,t) + log_beta(j,t) - logsumalphabeta(t) + ...
                log(weight(j,k)) + log_N_jkt(j,k,t) - log_mul_Gau(mean(:,:,j),var(:,:,j),obs(:,t), weight(j,:));
        end
    end
end
gamma = exp(log_gamma);

%% calculate sum of mean_numerator, var_numerator, aij_numerator and denominator (single data)
mean_numerator = zeros(dim,num_of_mix,num_of_state);
var_numerator = zeros(dim,num_of_mix,num_of_state);
wei_numerator = zeros(num_of_state,num_of_mix);
wei_denominator = zeros(num_of_state,1);
mean_var_denominator = zeros(num_of_state,num_of_mix);
aij_numerator = zeros(num_of_state,num_of_state);
for j = 2:num_of_state-1
    for k = 1:num_of_mix
        for t = 1:T
            mean_numerator(:,k,j) = mean_numerator(:,k,j) + gamma(j,k,t)*obs(:,t);
            %var_numerator(:,j) = var_numerator(:,j)+ gamma(j,t)*(obs(:,t)-mean(:,j)).^2;
            var_numerator(:,k,j) = var_numerator(:,k,j)+ gamma(j,k,t)*(obs(:,t)).*(obs(:,t));
            wei_numerator(j,k) = wei_numerator(j,k) + gamma(j,k,t);
            wei_denominator(j) = wei_denominator(j) + gamma(j,k,t);
            mean_var_denominator(j,k) = mean_var_denominator(j,k) + gamma(j,k,t);
        end
    end
end

for i = 2:num_of_state-1
    for j = 2:num_of_state-1
        for t = 1:T
            aij_numerator(i,j) = aij_numerator(i,j) + exp(log_Xi(i,j,t));
        end
    end
end

log_likelihood = log_alpha(num_of_state,T+1);
likelihood = exp(log_alpha(num_of_state,T+1));

end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function log_b = log_mul_Gau (mean_i, var_i, o_i, c_j)
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

function logsumalpha = log_sum_alpha(log_alpha_t,aij_j)
len_x = size(log_alpha_t,1);
y = -Inf(1,len_x);
ymax = -Inf;
for i = 1:len_x
    y(i) = log_alpha_t(i) + log(aij_j(i));
    if y(i) > ymax
        ymax = y(i);
    end
end
if ymax == Inf;
    logsumalpha = Inf;
else
    sum_exp = 0;
    for i = 1:len_x
        if ymax == -Inf && y(i) == -Inf
            sum_exp = sum_exp + 1;
        else
            sum_exp = sum_exp + exp(y(i) - ymax);
        end
    end
    logsumalpha = ymax + log(sum_exp);
end
end

function logsumbeta = log_sum_beta(aij_i,mean,var,obs,weight,beta_t1)
num_of_state = size(mean,3); % number of state
y = -Inf(1,num_of_state);
ymax = -Inf;
for j = 1:num_of_state
    y(j) = log(aij_i(j)) + log_mul_Gau(mean(:,:,j),var(:,:,j),obs,weight(j,:)) + beta_t1(j);
    if y(j) > ymax
        ymax = y(j);
    end
end
if ymax == Inf
    logsumbeta = Inf;
else
    sum_exp = 0;
    for i = 1:num_of_state
        if ymax == -Inf && y(i) == -Inf
            sum_exp = sum_exp + 1;
        else
            sum_exp = sum_exp + exp(y(i) - ymax);
        end
    end
    logsumbeta = ymax + log(sum_exp);
end
end

function logsumalphabeta = log_sum_alpha_beta(log_alpha, log_beta)
len_x = size(log_alpha,1); % number of state
y = -Inf(1,len_x);
ymax = -Inf;
for j = 1:len_x
    y(j) = log_alpha(j) + log_beta(j);
    if y(j) > ymax
        ymax = y(j);
    end
end
if ymax == Inf
    logsumalphabeta = Inf;
else
    sum_exp = 0;
    for i = 1:len_x
        if ymax == -Inf && y(i) == -Inf
            sum_exp = sum_exp + 1;
        else
            sum_exp = sum_exp + exp(y(i) - ymax);
        end
    end
    logsumalphabeta = ymax + log(sum_exp);
end
end
