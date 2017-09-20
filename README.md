# GMM HMM for isolated digits recognition applying EM algorithm
In this project we would like to deal with training GMM-HMM for isolated words data applying EM algorithm. The testing phase is also considered using Viterbi algorithm. The results showed the performances which obtained by Matlab programming are similar to HTK's ones. 

Before running these programs, please first prepare the training and testing data. Excerpts of TIDIGITS database can be obtained from this link:

http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/speech%20recognition%20course.html 

with the title of "isolated TI digits training files, 8 kHz sampled, endpointed: (isolated_digits_ti_train_endpt.zip)." 

or

you may download directly the .zip file of training database only from this link:

- training data:

http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/speech%20recognition%20course/databases/isolated_digits_ti_train_endpt.zip

- testing data:

http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/speech%20recognition%20course/databases/isolated_digits_ti_test_endpt.zip

Please decompress all the data sets, then locate training and testing data into directories 'wav\isolated_digits_ti_train_endpt' and 'wav\isolated_digits_ti_test_endpt', respectively. 

!!! Update: 2017-09-07

We have just added some feature extracting functions that would help you to convert '.wav' files to '.mfc' files (feature files)

Now you may run this project with only one click!

Please run the main function "EM_HMM_multiGaussian_isolated_digit_main.m" to start the work. 
The feature file format used in this version is compactable with the HTK format.
