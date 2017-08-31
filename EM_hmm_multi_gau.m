function HMM = EM_hmm_multi_gau(HMM, trainingfile)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

log_likelihood = 0;
likelihood = 0;
num_of_uter = size(trainingfile,1);
[DIM, num_of_mix, num_of_state, num_of_model] = size(HMM.mean); % N: number of states, NOT including START and END states (nodes) in HMM
[sum_mean_numerator, sum_var_numerator, sum_mean_var_denominator, sum_wei_numerator, sum_wei_denominator, sum_aij_numerator] = ...
    initial_sum_mean_var_aij(DIM, num_of_mix, num_of_state, num_of_model);

for u = 1:num_of_uter
    
    digit = trainingfile{u,1}; % digit: MODEL ID (0, 1, 2,..., 9)
    filename = trainingfile{u,2};
    
    mfcfile = fopen(filename, 'r', 'b' );
    if mfcfile ~= -1
        nSamples = fread(mfcfile, 1, 'int32');
        sampPeriod = fread(mfcfile, 1, 'int32')*1E-7;
        sampSize = fread(mfcfile, 1, 'int16');
        dim = 0.25*sampSize; % dim = 39
        parmKind = fread(mfcfile, 1, 'int16');
        
        features = fread(mfcfile, [dim, nSamples], 'float');
        
        [mean_numerator, var_numerator, mean_var_denominator, wei_numerator, wei_denominator, aij_numerator, log_likelihood_i, likelihood_i] =...
            forward_backward_hmm_mulgau_log_math(HMM.mean(:,:,:,digit), HMM.var(:,:,:,digit), HMM.Aij(:,:,digit), HMM.weight(:,:,digit), features); % model k_th
        
        sum_mean_numerator(:,:,:,digit) = sum_mean_numerator(:,:,:,digit) + mean_numerator(:,:,2:end-1);
        sum_var_numerator(:,:,:,digit) = sum_var_numerator(:,:,:,digit) + var_numerator(:,:,2:end-1);
        sum_mean_var_denominator(:,:,digit) = sum_mean_var_denominator(:,:,digit) + mean_var_denominator(2:end-1,:);        
        sum_wei_numerator(:,:,digit) = sum_wei_numerator(:,:,digit) + wei_numerator(2:end-1,:);
        sum_wei_denominator(:,digit) = sum_wei_denominator(:,digit) + wei_denominator(2:end-1);
        sum_aij_numerator(:,:,digit) = sum_aij_numerator(:,:,digit) + aij_numerator(2:end-1,2:end-1);
        
        log_likelihood = log_likelihood + log_likelihood_i;
        likelihood = likelihood + likelihood_i;
        
        fclose(mfcfile);        
    end
end

% calculate value of means, variances, weight, aij
for digit = 1:num_of_model
    for n = 1:num_of_state
        for k = 1:num_of_mix
            HMM.mean(:,k,n,digit) = sum_mean_numerator(:,k,n,digit) / sum_mean_var_denominator (n,k,digit);
            HMM.var (:,k,n,digit) = sum_var_numerator(:,k,n,digit) / sum_mean_var_denominator (n,k,digit) -  HMM.mean(:,k,n,digit).* HMM.mean(:,k,n,digit);
            HMM.weight(n,k,digit) = sum_wei_numerator(n,k,digit) / sum_wei_denominator (n,digit);
        end
    end
end
for digit = 1:num_of_model
    for i = 2:num_of_state+1
        for j = 2:num_of_state+1
            HMM.Aij (i,j,digit) = sum_aij_numerator(i-1,j-1,digit) / sum_wei_denominator (i-1,digit);
        end
    end
    HMM.Aij (num_of_state+1,num_of_state+2,digit) = 1 - HMM.Aij (num_of_state+1,num_of_state+1,digit);
end
HMM.Aij (num_of_state+2,num_of_state+2,digit) = 1;

end