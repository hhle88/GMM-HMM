function [sum_mean_numerator, sum_var_numerator, sum_mean_var_denominator, sum_wei_numerator, sum_wei_denominator, sum_aij_numerator] = ...
    initial_sum_mean_var_aij (DIM, num_of_mix, num_of_state, num_of_model)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization of value of sum_of_features, sum_of_features_square, num_of_feature, num_of_jump

    sum_mean_numerator = zeros(DIM, num_of_mix, num_of_state, num_of_model);
    sum_var_numerator = zeros(DIM, num_of_mix, num_of_state, num_of_model);
    sum_mean_var_denominator = zeros(num_of_state, num_of_mix, num_of_model);
    sum_wei_numerator = zeros(num_of_state, num_of_mix, num_of_model);
    sum_wei_denominator = zeros(num_of_state, num_of_state);
    sum_aij_numerator = zeros(num_of_state, num_of_state, num_of_model);    
end