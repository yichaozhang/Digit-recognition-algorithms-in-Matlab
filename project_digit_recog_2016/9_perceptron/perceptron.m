%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-09-07
%%% Description:
  % In this function, we will focus on the design of linear classifiers,
  % regardless of the underlying distribution describing the training data.
  % The major advantage of linear classifiers is their simplicity and
  % computational attributiveness.
  % Our aim is to compute the unknown parameters wi, i=0,1,...,m, defining
  % the decision hyperplane. 
  % The perceptron algorithm is iterative. The strategy starts with a
  % random guess at the weights w, and then iteratively change the weights
  % to move the hyperplane in a direction that lowers the classification
  % error.
%%% Input
  % digit_dimension: scalar. Set the dimension of each digit input after PCA;
  % lr: scalar. set the learning rate of the weighting update speed.
%%% Output
  % accuracy(scalar):The probability that a number is correctly detected.
  % weighting(digit_dimension*10 matrix): Weighting matrix after updated.
  % label_estimate (1*num_test matrix):Contains the estimated test labels.
%%% Notice, all training data and testing data are embeded. And the number
  % of trn and tst data can be tuned in function setup1. Since the
  % functions in this project are written in order to compare the digit
  % recognition accuracy of different algorithms. Thus, there is no need to
  % change the training and testing data set.

function [accuracy,weighting,label_estimate] = perceptron(digit_dimension,lr)
  
global main_folder trn_data_num test_data_num 

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
[trn_after_pca,~,PCA_info] = task1_PCA(data_trn,label_trn,digit_dimension,image);
trn_af_PCA   = trn_after_pca.image;                                         % Train data after PCA

  %%% 1.2: Load the test data, and use PCA results from train data to decrease 
  %%% the dimension of test data.(In this function, test data are automatically embeded.)
file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Check and open the Testing dataset.
if exist (file_to_open_testdata,'file')
    testdata=load('Testnumbers');                                              
else
    error('check the directory of training dataset again')
end
Testnumbers  = testdata.Testnumbers;
data_test    = Testnumbers.test_image_ex(:,1:test_data_num);                % Test data  (784*test_data_num)
label_test   = Testnumbers.test_label_ex(1:test_data_num,:)';               % Test label (1*test_data_num)
mean_trn     = PCA_info.mean_trn;                                           %(784*1 vec):contains mean of trn data of each pattern.
std_trn      = PCA_info.std_trn;                                            % Standard deviation of trn data of each pattern.
transformation_matrix = PCA_info.transformation_matrix;                     % Transformation matrix (digit_dimension*784):t_m*cov(trn_n')*t_m'=diagnal matrix which contains biggest NY.
data_test_normal = data_test;
std_index    = find(std_trn~=0);
for ii = 1:test_data_num                                                    % Normalize the testing data based on mean and variance of training data
    data_test_normal(std_index,ii) = (data_test(std_index,ii)-mean_trn(std_index))./std_trn(std_index);
end
test_af_PCA = transformation_matrix*data_test_normal;                       % Testing data after PCA (digit_dimension*trn_data_num)

  %%% 2nd step: Perceptron running
    % Principle: Define cost J(W,X)=alfa*(|W'X|-W'X). If the training data is correctly recognized, 
    % W'X>0, thus, J=0; If estimated label is different from the real one, then W'X<0. Thus, J<0. 
    % By using Gradient descent method, W(k+1)=W(k)-C*diff(J/W(k)). And we can get diff(J/W(k))=
    % 0.5*(sgn(W'X)-X), where sgn(W'X)=1 if W'X>0 and 0 else. Thus, W(k+1)=W(k) if W'X>0 and 
    % W(k+1)=W(k)+C*X(k) else.
weighting = 2*rand(digit_dimension,10)-1;                                   % Initialize weights matrix
for ii = 1:trn_data_num                                                     % Iteration for each train data
    d_x = weighting'*trn_af_PCA(:,ii);                                      % (10*1 vector)
    label_x = label_trn(ii);                                                % Label of input train data.
    [~,max_index] = max(d_x);                                               % Find out the position of the maximum value inside vector d_x.
    if (label_x ~= max_index-1)                                             % If di(x)>all other d values dj(x), then W(k+1)=W(k)
        for iii = 1:10                                                      % Else, Wi(k+1) = Wi(k)+lr*X(k)
            if iii ~= label_x+1                                             %       Wj(k+1) = Wj(k)-lr*X(k)
                weighting(:,iii) = weighting(:,iii) - lr*trn_af_PCA(:,ii);  
            else
                weighting(:,iii) = weighting(:,iii) + lr*trn_af_PCA(:,ii);
            end
        end
    end
end

  %%% 3rd step: Testing 
d_test         = weighting'*test_af_PCA;
sim_output     = compet(d_test);                                            % For each column of 'estimated_test_label_binary', find the position of maximum value and set it as 1, and all other data in this column as 0.
label_estimate = zeros(1,test_data_num);                                    % Transfer estimated label (10*1 format) into scalar (0-9 format) for each testing data.
for ii=1:test_data_num 
    num_index = find(sim_output(:,ii) == 1);
    label_estimate(1,ii) = num_index-1;
end                 

label_diff = label_test-label_estimate;
accuracy   = sum(label_diff(:)==0)/length(label_diff);                      % Number of data which are correctly recognized compared to all data

end
    




















