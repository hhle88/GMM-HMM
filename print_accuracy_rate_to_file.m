function print_accuracy_rate_to_file(accuracy_rate)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lee-Min Lee, Hoang-Hiep Le
% EE Department, Dayeh University
% version 1 (2017-08-31)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fileID = fopen('accuracy_rate.txt','a');
t = datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF AM');
fprintf(fileID,'\n============================================================\n');
fprintf(fileID,'%s\t\taccuracy rate: %f',t,accuracy_rate);
fclose(fileID);
end