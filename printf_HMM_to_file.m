function printf_HMM_to_file(HMM, save_file_name, file_list_of_model)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOTICE:
% 1. This function is to print all items of GMM-HMM on HTK style and save them into the given "save_file_name"
% 2. Content of "file_list_of_model" can be modified by user. For example, if user is running this project 
%    for English digits, the content should be "one," "two,"... "nine". 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin==0
    load 'models\HMM_0.mat';
    save_file_name = 'HMM_GMM.txt';
    file_list_of_model = 'list_of_models.txt';
end

model = importdata(file_list_of_model);
[DIM, num_of_mix, num_of_state, num_of_model] = size(HMM.mean);
fileID = fopen(save_file_name,'w');

% fprintf(fileID,'<NUMBER OF MODEL> %d\n',num_of_model);
for k = 1:num_of_model
    fprintf(fileID,'~h "%s"\n<BEGINHMM>\n', model{k});
    fprintf(fileID,'<NUMSTATES> %d\n',num_of_state+2);
    for i = 1:num_of_state
        fprintf(fileID,'<STATE> %d\n', i+1);
        if (num_of_mix > 1)
            fprintf(fileID,'<NUMBMIXS> %d\n',num_of_mix);
        end
        for m = 1:num_of_mix
            %%% print mixture
            if (num_of_mix > 1)
                fprintf(fileID,'<MIXTURE> %d\t%f\n',m,HMM.weight(i,m,k));
            end
            %%% print mean
            fprintf(fileID,'<MEAN> %d\n',DIM);
            for d = 1:DIM
                fprintf(fileID,'%f  ',HMM.mean(d,m,i,k));
            end
            fprintf(fileID,'\n');
            %%% print variance
            fprintf(fileID,'<VARIANCE> %d\n',DIM);
            for d = 1:DIM
                fprintf(fileID,'%f  ',HMM.var(d,m,i,k));
            end
            fprintf(fileID,'\n');
        end
        fprintf(fileID,'\n');
    end
    %% print aij matrix
    fprintf(fileID,'<TRANSP> %d\n',num_of_state+2);
    for i = 1:num_of_state+2
        for j = 1:num_of_state+2
            fprintf(fileID,'%f  ',HMM.Aij(i,j,k));
        end
        fprintf(fileID,'\n');
    end
    fprintf(fileID,'\n');
end

fclose(fileID);
end