%%% Author:         Yichao Zhang
%%% Version:        1.0 (2015-08-20)
%%% Version:        2.0
%%% Date:           2015-09-13
%%% Description
  % This function implements the Bayesian classifier based on grayscale image.
  % Bayesian classifier is an algorithm for supervised learning that provides
  % a probability summary for each class/label based on the assumption that 
  % attributes given the classes are conditionally independent from each other.
  % For each test data, Bayes'rule is used to compute the posterior probability 
  % of each class given this data and select the class which has the
  % highest probability.
  % The differences between this function and function 'Bayesian_Binary' is
  % different way of calculating likelihood function. In this function, we
  % assume likelihood as Gaussian distribution. 
%%% Input
  % image_dimension (scalar): set the dimension of train/test data.
%%% Output
  % Posterior_P(TX*10):TX is the number of test data, and 10 is the number
  %   of classes(0-9). Each row contains posterior probability of one sample
  %   given different labels (from 0-9).The class which contains the highest
  %   posterior probability value is the class this data belongs.
  % label_estimate(1*TX):Contains the estimated labels after Bayesian
  %   classifier.
  % accuracy(scalar):The probability that a number is correctly detected.
%%% Notice, all training data and testing data are embeded. And the number
  % of trn and tst data can be tuned in function setup1. Since the
  % functions in this project are written in order to compare the digit
  % recognition accuracy of different algorithms, there is no need to
  % change the training and testing data set.

function [Posterior_P,label_estimate,accuracy]=Bayesian_Normal(image_dimension)

global main_folder trn_data_num test_data_num

%%% 1st Load: Bayesian based on gray image
% 1.1 Load train dataset
file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data=load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
data_trn    =Trainnumbers.trn_image_ex(:,1:trn_data_num);                   % Extract training dataset (784*NX)
label_trn   =Trainnumbers.trn_label_ex(1:trn_data_num,:)';                  % Extract training label (1*NX)
% 1.2 PCA
image       = 0;
[trn_after_pca,~,PCA_info]=task1_PCA(data_trn,label_trn,image_dimension,image);
% 1.3 Load test dataset
file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Check and open the Testing dataset.
if exist (file_to_open_testdata,'file')
    testdata=load('Testnumbers');                                              
else
    error('check the directory of training dataset again')
end
Testnumbers = testdata.Testnumbers;
data_test = Testnumbers.test_image_ex(:,1:test_data_num);
label_test = Testnumbers.test_label_ex(1:test_data_num,:)';
% 1.4 Normalization and PCA on testing dataset
mean_trn         = PCA_info.mean_trn;                                       %(784*1 vec):contains mean of trn data of each pattern.
std_trn          = PCA_info.std_trn;                                        % Standard deviation of trn data of each pattern.
std_index        = find(std_trn~=0);
data_test_normal = data_test;
for ii = 1:test_data_num                           % Normalize the testing data based on mean and variance of training data
    data_test_normal(std_index,ii) = (data_test(std_index,ii)-mean_trn(std_index))./std_trn(std_index);
end
transformation_matrix = PCA_info.transformation_matrix;                     % Transformation matrix (NY*784):t_m*cov(trn_n')*t_m'=diagnal matrix which contains biggest NY.
test_af_PCA = transformation_matrix*data_test_normal;                       % Testing data after PCA (NY*TX)
trn_af_PCA  = trn_after_pca.image;                                          % Train data after PCA
trn_label   = trn_after_pca.label;                                          % Train labels

%%% 2nd Baysian algorithm
% 2.1 Prior probability P(w_i)
prior_P = zeros(10,2);                                                      % 10*2 matrix. First column contains the digit (0-9), 
for ii = 1:length(prior_P)                                                  % second column contains the prior probability of each digit.
    prior_P(ii,1) = ii-1;                                                   % Digit 0-9
    prior_P(ii,2) = length(find(trn_label == ii-1))/trn_data_num;                     % Prior probability of corresponding digit.
end

% Since the principle of Bayesian classifier is to calculate the posterior
% probability of each class given observation and find the maximum one.
% And, P(wi|X)=P(X|wi)*P(wi)/P(X),proportional to P(X|wi)*P(wi),proportional
% to ln(P(X|wi))+ln(P(wi)). In this algorithm, assume the likelihood has
% Gaussian distribution.
% P(X|wi)=1/((2pi)^(n/2)*sqrt(det(cov)))*exp(-0.5*(X-mu)'inv(cov)*(X-mu))
% ln(P(X|wi))= -0.5*(X-mu)'inv(cov)*(X-mu)-0.5*ln(det(cov)).

% 2.2 Calculate the covariance matrix
cov_all_label       = [];                                                   % Initialize covariance matrix of all classes. 10*1 cell. Each cell contains cov matrix of one class.
mean_data_all_label = [];                                                   % Initialize mean value of data in each class. 10*1 cell.
det_cov_all_label   = [];                                                   % Determinant of cov matrix of all classes. 10*1 cell.
for ii = 1:length(prior_P)                                             
    data_per_label              = trn_af_PCA(:,trn_label==ii-1);            % Extract data which contains same label/class.
    mean_data_all_label{ii,1}   = mean(data_per_label,2);                     
    cov_all_label{ii,1}         = cov(data_per_label');                     % NY*NY matrix, covariance matrix of one/each class.
    det_cov_all_label{ii,1}     = det(cov_all_label{ii,1});
end
% 2.3 Posterior probability
Posterior_P = zeros(length(label_test),length(prior_P));                    % num_test_data*10 matrix. Each row contains posterior probability of one sample given different labels (from 0-9)
for ii = 1:size(test_af_PCA,2)                                              % Iteration for extracting each test data
    for jj = 1:length(prior_P)                                              % Iteration for classes/labels
        cov_per_label       = cov_all_label{jj,1};                          % NY*NY matrix, covariance matrix of one/each class.
        mean_data_per_label = mean_data_all_label{jj,1};                    % Mean of observation data of one/each class.
        det_cov_per_label   = det_cov_all_label{jj,1};                      % Determinant of covariance of a class.
        h_x = -0.5*(test_af_PCA(:,ii)-mean_data_per_label)'*inv(cov_per_label)*(test_af_PCA(:,ii)-mean_data_per_label)+log(prior_P(jj,2))-0.5*log(det_cov_per_label);
        Posterior_P(ii,jj)  = h_x;                                          % Calculate and store the posterior probability
    end                                                                     % Posterior probability P(wi|X)
end

% 3rd Find the label of testing data
[~,index_label] = max(Posterior_P,[],2);                                    % The class which contains the highest posterior
label_estimate = index_label-1;                                             % probability value is the class this data belongs.
label_diff = label_test-label_estimate';
accuracy = sum(label_diff(:)==0)/length(label_diff);                        % Number of data which are correctly recognized compared to all data

end

