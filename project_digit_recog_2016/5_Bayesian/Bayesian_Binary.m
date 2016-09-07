%%% Author:         Yichao Zhang
%%% Version:        1.0 (2015-08-19)
%%% Version:        2.0
%%% Date:           2015-09-13
%%% Description
  % This function implements the Bayesian classifier based on binary image.
  % Bayesian classifier is an algorithm for supervised learning that provides
  % a probability summary for each class/label based on the assumption that 
  % attributes given the classes are conditionally independent from each other.
  % For each test data, Bayes'rule is used to compute the posterior probability 
  % of each class given this data and select the class which has the
  % highest probability.
  % In this function, we first calculate prior probability, then calculate
  % the probability that, among all data in one label, the jth pattern of
  % Xi equals to 1. Next to that, likelihood P(X|wi) is calculated, which
  % P(X|wi)=P[X=(x_1,x_2,...,x_784)|wi]=prod(P(Xj=0or1|wi)). Finally,
  % posterior probability function is get. The other function 'Bayesian_normal'
  % assume Gaussian distribution for its likehood function.
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

function [Posterior_P,label_estimate,accuracy]=Bayesian_Binary()

global main_folder trn_data_num test_data_num

%%% 1st Load: Bayesian based on binary image
% 1.1 Load train dataset
file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data=load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
data_trn_binary=Trainnumbers.trn_image_binary(:,1:trn_data_num);            % Extract training dataset (784*NX)
label_trn=Trainnumbers.trn_label_ex(1:trn_data_num,:)';                     % Extract training label (1*NX)
% 1.2 Load test dataset
file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Check and open the Testing dataset.
if exist (file_to_open_testdata,'file')
    testdata=load('Testnumbers');                                              
else
    error('check the directory of training dataset again')
end
Testnumbers = testdata.Testnumbers;
data_test = Testnumbers.test_image_binary(:,1:test_data_num);
label_test = Testnumbers.test_label_ex(1:test_data_num,:)';

%%% 2nd Bayesian algorithm
% 2.1 Prior probability P(w_i)
prior_P = zeros(10,2);                                                      % 10*2 matrix. First column contains the digit (0-9), 
for ii = 1:length(prior_P)                                                  % second column contains the prior probability of each digit.
    prior_P(ii,1) = ii-1;                                                   % Digit 0-9
    prior_P(ii,2) = length(find(label_trn == ii-1))/trn_data_num;           % Prior probability of corresponding digit.
end

% 2.2 Calculate P_j(w_i)    (i=1:10  10 classes/digits in this case) (j=1:784 is the number of pixels/patterns for each observation,)
% P_j(w_i) means: among all data in the label i (we call the dataset inside
% label i Xi), calculate the probability that the jth pattern of Xi equals to 1.   
Pj_wi = zeros(size(data_trn_binary,1),length(prior_P));
for ii = 1:length(prior_P)
    data_per_label = data_trn_binary(:,label_trn == ii-1);
    for jj = 1:size(data_trn_binary,1)
        Pj_wi(jj,ii) = (sum(data_per_label(jj,:))+1)/(size(data_per_label,2)+2);
    end
end

% 2.3 Likelihood function P(X|wi)
% P(X|wi)=P[X=(x_1,x_2,...,x_784)|wi]=prod(P(Xj=0or1|wi))
likelihood_P = zeros(length(label_test),length(prior_P));                   % num_test_data*10 matrix. Each row contains likelihood probability of one sample given different classes (from 0-9)
for ii = 1:size(data_test,2)
    for jj = 1:length(prior_P)
        sample = data_test(:,ii);                                           % Pick up one sample
        index_0 = find(sample == 0);                                        % Find the index where pixel inside is 0
        Pj_wi_vec = Pj_wi(:,jj);                                            % Since Pj(wi) calculates the probability that the jth pattern of Xi equals to 1.   
        Pj_wi_vec(index_0) = ones(length(index_0),1)-Pj_wi_vec(index_0);    % P(Xj=0|wi)=1-Pj(wi),P(Xj=1|wi)=Pj(wi).
        likelihood_P(ii,jj) = prod(Pj_wi_vec);                              % Product:P(X|wi)=P[X=(x_1,x_2,...,x_784)|wi]=prod(P(Xj=0or1|wi))                       
    end
end
    
% 2.4 Posterior probability P(wi|X)
Posterior_P = zeros(length(label_test),length(prior_P));                    % num_test_data*10 matrix. Each row contains posterior probability of one sample given different classes (from 0-9)
for ii = 1:size(data_test,2)
    for jj = 1:length(prior_P)
        Posterior_P(ii,jj) = (prior_P(jj,2)*likelihood_P(ii,jj))/sum(prior_P(:,2).*likelihood_P(ii,:)');
    end  
end

%%% Step3 Find the label of testing data
[~,index_label] = max(Posterior_P,[],2);                                    % The class which contains the highest posterior
label_estimate = (index_label-1)';                                          % probability value is the class this data belongs.
label_diff = label_test-label_estimate;
accuracy = sum(label_diff(:)==0)/length(label_diff);                        % Number of data which are correctly recognized compared to all data

end



