function generate_trainingfile_list
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generating a file contains list of paths of training data (*.mfc or *.mfcc files)

clear trainingfile_list.mat;
training_file_list = 'trainingfile_list.mat';
fea_dir = 'mfcc';
k = 0;
for PHASE = 1:6
    for MODEL = 0:9
        for spk = 1:2:99
            k = k + 1;
            trainingfile{k,1} = MODEL+1;
            trainingfile{k,2} = sprintf('%s\\S%d\\%02d_%02d.mfc',fea_dir,PHASE,spk,MODEL);
        end
    end
end
save(training_file_list, 'trainingfile');
end