
%% Bayesian 
global main_folder
file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data=load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
NX = 60000;
data_trn_binary=Trainnumbers.trn_image_binary(:,1:NX);                      % Extract training dataset (784*NX)
label_trn=Trainnumbers.trn_label_ex(1:NX,:)';                               % Extract training label (1*NX)
NY = 100;                                                                   % Number of patterns want to keep
image = 0;
[trn_after_pca,reconstruction,PCA_info]=task1_PCA(data_trn_binary,label_trn,NY,image);

file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Check and open the Testing dataset.
if exist (file_to_open_testdata,'file')
    testdata=load('Testnumbers');                                              
else
    error('check the directory of training dataset again')
end
Testnumbers = testdata.Testnumbers;
TX = 3000;
data_test = Testnumbers.test_image_ex(:,1:TX);
data_test2 = mat2gray(data_test);
label_test = Testnumbers.test_label_ex(1:TX,:)';
%% Baysian based on binary image
% Prior probability P(w_i)
prior_P = zeros(10,2);                                                      % 10*2 matrix. First column contains the digit (0-9), 
for ii = 1:length(prior_P)                                                  % second column contains the prior probability of each digit.
    prior_P(ii,1) = ii-1;                                                   % Digit 0-9
    prior_P(ii,2) = length(find(label_trn == ii-1))/NX;                     % Prior probability of corresponding digit.
end

% Calculate P_j(w_i)    (i=1:10  10 labels/digits in this case) (j=1:784 is the number of pixels/patterns for each observation,)
% P_j(w_i) means: among all data in the label i (we call the dataset inside
% label i Xi), calculate the probability that the jth pattern of Xi equals to 1.   
Pj_wi = zeros(size(data_trn_binary,1),length(prior_P));
for ii = 1:length(prior_P)
    data_per_label = data_trn_binary(:,find(label_trn == ii-1));
    for jj = 1:size(data_trn_binary,1)
        Pj_wi(jj,ii) = (sum(data_per_label(jj,:))+1)/(size(data_per_label,2)+2);
    end
end

% Likelihood function P(X|wi)
% P(X|wi)=P[X=(x_1,x_2,...,x_784)|wi]=prod(P(Xj=0or1|wi))
likelihood_P = zeros(length(label_test),length(prior_P));                   % num_test_data*10 matrix. Each row contains likelihood probability of one sample given different labels (from 0-9)
for ii = 1:size(data_test,2)
    for jj = 1:length(prior_P)
        sample = data_test(:,ii);                                           % Pick up one sample
        index_0 = find(data_test(:,ii) == 0);                               % Find the index where pixel inside is 0
        Pj_wi_vec = Pj_wi(:,jj);                                            % Since Pj(wi) calculates the probability that the jth pattern of Xi equals to 1.   
        Pj_wi_vec(index_0) = ones(length(index_0),1)-Pj_wi_vec(index_0);    % P(Xj=0|wi)=1-Pj(wi),P(Xj=1|wi)=Pj(wi).
        likelihood_P(ii,jj) = prod(Pj_wi_vec);                              % Product:P(X|wi)=P[X=(x_1,x_2,...,x_784)|wi]=prod(P(Xj=0or1|wi))                       
    end
end
    
% Posterior probability
Posterior_P = zeros(length(label_test),length(prior_P));                    % num_test_data*10 matrix. Each row contains posterior probability of one sample given different labels (from 0-9)
for ii = 1:size(data_test,2)
    for jj = 1:length(prior_P)
        Posterior_P(ii,jj) = (prior_P(jj,2)*likelihood_P(ii,jj))/sum(prior_P(:,2).*likelihood_P(ii,:)');
    end
end

% Find the label of testing data
[~,index_label] = max(Posterior_P,[],2);
label_estimate = index_label-1;
label_diff = label_test-label_estimate';
accuracy = sum(label_diff(:)==0)/length(label_diff);                        % Number of data which are correctly recognized compared to all data


%% Bayesian normal
file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data=load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
NX = 20000;
data_trn=Trainnumbers.trn_image_ex(:,1:NX);                          % Extract training dataset (784*NX)
label_trn=Trainnumbers.trn_label_ex(1:NX,:)';                               % Extract training label (1*NX)
NY = 100;                                                                   % Number of patterns want to keep
image = 0;
[trn_after_pca,reconstruction,PCA_info]=task1_PCA(data_trn,label_trn,NY,image);

file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Check and open the Testing dataset.
if exist (file_to_open_testdata,'file')
    testdata=load('Testnumbers');                                              
else
    error('check the directory of training dataset again')
end
Testnumbers = testdata.Testnumbers;
TX = 3000;
data_test = Testnumbers.test_image_ex(:,1:TX);
label_test = Testnumbers.test_label_ex(1:TX,:)';

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
trn_af_PCA = trn_after_pca.image;                                           % Train data after PCA
trn_label = trn_after_pca.label;                                            % Train labels
test_label = label_test;

% Prior probability P(w_i)
prior_P = zeros(10,2);                                                      % 10*2 matrix. First column contains the digit (0-9), 
for ii = 1:length(prior_P)                                                  % second column contains the prior probability of each digit.
    prior_P(ii,1) = ii-1;                                                   % Digit 0-9
    prior_P(ii,2) = length(find(trn_label == ii-1))/NX;                     % Prior probability of corresponding digit.
end

%
% Since the principle of Bayesian classifier is to calculate the posterior
% probability of each class given observation and find the maximum one.
% And, P(wi|X)=P(X|wi)*P(wi)/P(X),proportional to P(X|wi)*P(wi),proportional
% to ln(P(X|wi))+ln(P(wi)). In this algorithm, assume the likelihood has
% Gaussian distribution.
% P(X|wi)=1/((2pi)^(n/2)*sqrt(det(cov)))*exp(-0.5*(X-mu)'inv(cov)*(X-mu))
% ln(P(X|wi))= -0.5*(X-mu)'inv(cov)*(X-mu)-0.5*ln(det(cov)).

% Calculate the covariance matrix
cov_all_label       = [];                                                   % Initialize covariance matrix of all classes. 10*1 cell. Each cell contains cov matrix of one class.
mean_data_all_label = [];                                                   % Initialize mean value of data in each class. 10*1 cell.
det_cov_all_label   = [];                                                   % Determinant of cov matrix of all classes. 10*1 cell.
for ii = 1:length(prior_P)                                             
    data_per_label = trn_af_PCA(:,find(trn_label==ii-1));                   % Extract data which contains same label/class.
    mean_data_all_label{ii,1} = mean(data_per_label,2);                     
    cov_all_label{ii,1} = cov(data_per_label');                             % NY*NY matrix, covariance matrix of one/each class.
    det_cov_all_label{ii,1} = det(cov_all_label{ii,1});
end

Posterior_P = zeros(length(label_test),length(prior_P));                    % num_test_data*10 matrix. Each row contains posterior probability of one sample given different labels (from 0-9)
for ii = 1:size(test_af_PCA,2)                                              % Iteration for extracting each test data
    for jj = 1:length(prior_P)                                              % Iteration for classes/labels
        cov_per_label = cov_all_label{jj,1};                                % NY*NY matrix, covariance matrix of one/each class.
        mean_data_per_label = mean_data_all_label{jj,1};                    % Mean of observation data of one/each class.
        det_cov_per_label = det_cov_all_label{jj,1};                        % Determinant of covariance of a class.
        h_x = -0.5*(test_af_PCA(:,ii)-mean_data_per_label)'*inv(cov_per_label)*(test_af_PCA(:,ii)-mean_data_per_label)+log(prior_P(jj,2))-0.5*log(det_cov_per_label);
        Posterior_P(ii,jj) = h_x;                                           % Calculate and store the posterior probability
    end                                                                     % Posterior probability P(wi|X)
end

% Find the label of testing data
[~,index_label] = max(Posterior_P,[],2);                                    % The class which contains the highest posterior
label_estimate = index_label-1;                                             % probability value is the class this data belongs.
label_diff = label_test-label_estimate';
accuracy = sum(label_diff(:)==0)/length(label_diff);                        % Number of data which are correctly recognized compared to all data




