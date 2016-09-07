%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-08-17
%%% Description
  % K nearest neighbourhood: If one data is selected, calculate the
  % distance between this data with all other training data. Here,
  % Euclidean distance is used. Afterwards, choose the K smallest data and
  % see their labels. The most freqency label is the one test data belongs.
%%% INPUT
  % data_test(784*TX):TX is the number of testing data, 784 is the number
  %   of pixels/pattern for each observation;
  % label_test(1*TX):Labels of all observations are included.
  % K(scalar):K is the number of nearest data to each testing data.
  % NY(scalar):Number of patterns we want to keep after PCA.
%%% OUTPUT
  % test_label_estimate(1*TX):Labels we get after KNN algorithm is
  % implemented.
  % accuracy(scalar):The probability that a number is correctly detected.
  
function [test_label_estimate,accuracy]=KNN_MNIST(data_test,label_test,NY,K)

global main_folder 

file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data=load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
NX = 50000;                                                                 % Number of training data selected
data_trn=Trainnumbers.trn_image_ex(:,1:NX);                                 % Extract training dataset (784*NX)
label_trn=Trainnumbers.trn_label_ex(1:NX,:)';                               % Extract training label (1*NX)
image = 0;
[trn_after_pca,reconstruction,PCA_info]=task1_PCA(data_trn,label_trn,NY,image);
TX = length(label_test);                                                    % Number of Testing data
mean_trn = PCA_info.mean_trn;                                               %(784*1 vec):contains mean of trn data of each pattern.
std_trn = PCA_info.std_trn;                                                 % Standard deviation of trn data of each pattern.
std_index = find(std_trn~=0);
data_test_normal = data_test;
for ii = 1:TX                           % Normalize the testing data based on mean and variance of training data
    data_test_normal(std_index,ii) = (data_test(std_index,ii)-mean_trn(std_index))./std_trn(std_index);
end
transformation_matrix = PCA_info.transformation_matrix;                     % Transformation matrix (NY*784):t_m*cov(trn_n')*t_m'=diagnal matrix which contains biggest NY.
test_af_PCA = transformation_matrix*data_test_normal;                       % Testing data after PCA (NY*TX)
trn_image = trn_after_pca.image;                                            % Train data after PCA
trn_label = trn_after_pca.label;                                            % Train labels
test_label_estimate = [];                                                   % Contains estimated labels after KNN
for ii = 1:length(test_af_PCA)
    norm_data = sum((trn_image-repmat(test_af_PCA(:,ii),1,length(trn_image))).^2);  % Square Eucledian distance of one test data with all train data
    [B,Index_KNN] = sort(norm_data,'ascend');                               % Sort the distances from small to big
    label_KNN_list = trn_label(Index_KNN(1:K));                             % Find K smallest distance and check the corresponding trn labels
    test_label_estimate(1,ii) = mode(label_KNN_list);                       % Find the most frequent numbers appear in these labels. And this number is the label of this data
end
label_diff = label_test-test_label_estimate;
accuracy = sum(label_diff(:)==0)/length(label_diff);                        % Number of data which are correctly recognized compared to all data

end

% Non-parametric classifiers have several very important advantages that
% are not shared by most learning-based approaches: 1)Can naturally handle
% a huge number of classes; 2)Avoid overfitting of parameters; 3)Require no
% learning/training phase.



