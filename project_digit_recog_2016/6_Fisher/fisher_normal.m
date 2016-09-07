%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-09-13
%%% Description:
  % Another way to view linear discriminants: find the 1D subspace that
  % maximizes the separation between two classes.
%%% Input
  % digit_dimension: scalar. Set the dimension of each digit input after PCA;
%%% Output
  % accuracy(scalar):The probability that a number is correctly detected.
%%% Notice, all training data and testing data are embeded. And the number
  % of trn and tst data can be tuned in function setup1. Since the
  % functions in this project are written in order to compare the digit
  % recognition accuracy of different algorithms. Thus, there is no need to
  % change the training and testing data set.

function [accuracy,label_estimate] = fisher_normal (digit_dimension)

global main_folder trn_data_num test_data_num 
global in_class_mean trn_class_complete Cov_in_class

  %%% 1st step: Pre-processing
  %%% 1.1. Load the train data, and use PCA to decrease the dimension.
  %%% (In this function, train data are automatically embaded.)
file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data = load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
data_trn     = Trainnumbers.trn_image_ex(:,1:trn_data_num);                 % Extract training dataset (784*trn_data_num)
label_trn    = Trainnumbers.trn_label_ex(1:trn_data_num,:)';                % Extract training labels (1*trn_data_num)
image        = 0;
[trn_after_pca,~,PCA_info] = task1_PCA(data_trn,label_trn,digit_dimension,image); % PCA
trn_af_PCA   = trn_after_pca.image;                                         % Train data after PCA

  %%% 1.2: Load the test data, and use PCA results from train data to decrease 
  %%% the dimension of test data.(In this function, test data are automatically embeded.)
file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Check and open the Testing dataset.
if exist (file_to_open_testdata,'file')
    testdata = load('Testnumbers');                                              
else
    error('check the directory of training dataset again')
end
Testnumbers  = testdata.Testnumbers;
data_test    = Testnumbers.test_image_ex(:,1:test_data_num);                % Test data  (784*test_data_num)
label_test   = Testnumbers.test_label_ex(1:test_data_num,:)';               % Test label (1*test_data_num)
mean_trn     = PCA_info.mean_trn;                                           %(784*1 vec):contains mean of trn data of each pattern.
std_trn      = PCA_info.std_trn;                                            % Standard deviation of trn data of each pattern.
transformation_matrix = PCA_info.transformation_matrix;                     % Transformation matrix (digit_dimension*784):t_m*cov(trn_n')*t_m'=diagnal matrix which contains biggest NY.
data_test_normal      = data_test;
std_index    = find(std_trn~=0);
for ii = 1:test_data_num                                                    % Normalize the testing data based on mean and variance of training data
    data_test_normal(std_index,ii) = (data_test(std_index,ii)-mean_trn(std_index))./std_trn(std_index);
end
test_af_PCA = transformation_matrix*data_test_normal;                       % Testing data after PCA (digit_dimension*trn_data_num)

  %%% 1.3 Extract data into classes 0-9.
trn_class_complete  = cell(10,1);                                           % Each cell contains all digits belonging to one label.
in_class_mean       = zeros(digit_dimension,10);                            % 10 columns contain mean value of all data belongs to individual label.
Cov_in_class        = cell(10,1);                                           % Covariance matrix of data in each class (within class scatter)
for ii = 1:10
    class_index     = find(label_trn == ii-1);                              % Find the position of data which contains same label.
    trn_per_class   = trn_af_PCA(:,class_index);                            % Extract train data with same label.
    trn_class_complete{ii,1} = trn_per_class;                               % Store the train data per label into matrix 'trn_class_complete'.
    in_class_mean(:,ii)      = mean(trn_per_class')';                       % within class mean
    Cov_in_class{ii,1}       = cov(trn_per_class');                         % within class scatter    
end  

class_num = zeros(10,size(test_af_PCA,2));
for class1 = 2:10
    for class2 = 1:(class1-1)
        %%% 2nd step: Fisher classifier
        classfit = fisher_classify(class1,class2,test_af_PCA);
%%% INPUT
  % class1 (scalar): shows the value of first label (0-9)
  % class2 (scalar): shows the value of second label (0-9)
  % tst_data (dimension*num_test_data): test data inputs.
%%% OUTPUT
  % classfit (1*num_test_data matrix): shows the classification reusults of
  % each testing data (compare class1 and class2, choose the one which is 
  % more similar to each test data)

        %%% 3rd step: Test classification results
        for ii = 1:length(classfit)
            class_num(classfit(ii),ii) = class_num(classfit(ii),ii)+1;
        end
    end
end

sim_output     = compet(class_num);                                         % For each column of 'class_num', find the position of maximum value and set it as 1, and all other data in this column as 0.
label_estimate = zeros(1,test_data_num);                                    % Transfer estimated label (10*1 format) into scalar (0-9 format) for each testing data.
for ii=1:test_data_num 
    num_index            = find(sim_output(:,ii) == 1);
    label_estimate(1,ii) = num_index-1;
end                 

label_diff = label_test-label_estimate;
accuracy   = sum(label_diff(:)==0)/length(label_diff);                      % Number of data which are correctly recognized compared to all data

end
        
        
        
        



