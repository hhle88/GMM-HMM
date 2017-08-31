function save_HMM_to_a_file(HMM, iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path_file = sprintf('models\\HMM_%d.mat',iter);
save(path_file, 'HMM');
save_file_name = sprintf('models\\GMM_HMM_%d.txt',iter);
file_list_of_model = 'list_of_models.txt';
printf_HMM_to_file(HMM, save_file_name, file_list_of_model);

end