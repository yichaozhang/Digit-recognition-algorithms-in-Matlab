%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-08-16
%%% Description:
%%% Authors:        yichao, ..., ... 

% addpath yourself
global main_folder data_folder KNN_result_str test_data_num
%% Pre-processing
% Extract data from raw data
trn_data_generate()
test_data_generate()

file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data=load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
NX = 8000;
data_trn=Trainnumbers.trn_image_ex(:,1:NX);                                 % Extract training dataset (784*NX)
label_trn=Trainnumbers.trn_label_ex(1:NX,:)';                               % Extract training label (1*NX)
NY = 100;                                                                   % Number of patterns want to keep
image = 0;
[trn_after_pca,reconstruction,PCA_info]=task1_PCA(data_trn,label_trn,NY,image);
%%% Description:  principal component analysis ("PCA") rotates the
%%% original data to new coordinates, making the data as "flat" as
%%% possible.PCA is useful for data reduction, noise rejection,
%%% visualization and data compression among other things. 
%% KNN
KNN_result_str = 'KNN_result.mat';
test_data_num = 500;
KNN_perform_compare();
%%% Description:    This function shows the results and accuracy of KNN
  % when varying the value of K and number of patterns kept after using
  % PCA.
KNN_results();

%% Bayesian classifier
  % Bayesian classifier is an algorithm for supervised learning that provides
  % a probability summary for each class/label based on the assumption that 
  % attributes given the classes are conditionally independent from each other.
  % For each test data, Bayes'rule is used to compute the posterior probability 
  % of each class given this data and select the class which has the
  % highest probability.
image_dimension = 100;
[Posterior_P,label_estimate,accuracy]=Bayesian_Binary();
  % This function implements the Bayesian classifier based on binary image.
[Posterior_P_normal,label_estimate_normal,accuracy_normal]=Bayesian_Normal(image_dimension);
  % This function implements the Bayesian classifier based on grayscale image.
%% Linear Classifier
digit_dimension = 100;
lr = 0.1;
[accuracy_perceptron,~,label_estimate_perceptron] = perceptron(digit_dimension,lr);
  % The perceptron algorithm is iterative. The strategy starts with a random guess at the weights w, and then  
  % iteratively change the weights to move the hyperplane in a direction that lowers the classification error.  
  % Principle: Define cost J(W,X)=alfa*(|W'X|-W'X). If the training data is correctly recognized, W'X>0, thus, J=0; 
  % If estimated label is different from the real one, then W'X<0. Thus, J<0. By using Gradient descent method, 
  % W(k+1)=W(k)-C*diff(J/W(k)). And we can get diff(J/W(k))=0.5*(sgn(W'X)-X), where sgn(W'X)=1 
  % if W'X>0 and 0 else. Thus, W(k+1)=W(k) if W'X>0 and W(k+1)=W(k)+C*X(k) else.  
[accuracy_fisher,label_estimate_fisher] = fisher_normal (digit_dimension);
  % Another way to view linear discriminants: find the 1D subspace that maximizes the separation between two   
  % classes.Suppose two classes of observations have means mu_0 and mu_1 and covariances V_0 and V_1. Then the 
  % linear combination of features w'*x will have means w'*mu_i and variance w'*V_i*w for i=0,1. Fisher defined
  % the separation between these two distributions to be the ratio of the variance between the classes to the 
  % variance within the classes: J_F = W_opt'*Cov_between_class*W_opt/(W_opt'*Cov_in_class_sum*W_opt);  
  % This measure is, in some sense, a measure of the signal-to-noise ratio for the class labelling. It can be
  % shown that the maximum separation occurs when W_opt proportional to inv(Cov_in_class_sum)*(mu_1-mu_0).   
  % Afterwards, W_opt is used to project the train data in two labels into one-dimensional plane: y.
  % y_thres is set to distinguish training data projecting on y plane.
  %%
  digit_dimension = 90;
  sigma = 15;
  [accuracy_fisher_kernel,label_estimate_kernel_fisher] = fisher_kernel (digit_dimension,sigma);
  % In statistics, Kernel Fisher discriminant analysis (KFD), is a kernel version of linear discriminant analysis.
  % Using the Kernel trick, LDA is implicitly performed in a new feature space, which allows non-linear mappings.  
  % Intuitively, the idea of LDA is to find a projection where class separation is maximized. To extend LDA to 
  % non-linear mappings, F, via some function theta.    
  % Notice, Kernel Fisher Discriminant shows great performance when the number of training data is small. 
  % When it is big, then running speed is too slow.
  
%% Neural Network

Back_Propagation;








