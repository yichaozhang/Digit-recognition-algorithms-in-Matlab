
tic
digit_dimension = 90;
sigma = 15;

global main_folder trn_data_num test_data_num 

  %%% 1st step: Pre-processing
  %%% 1.1. Load the train data, and use PCA to decrease the dimension.
  %%% (In this function, train data are automatically embaded.)
file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data=load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
data_trn    =Trainnumbers.trn_image_ex(:,1:trn_data_num);                   % Extract training dataset (784*trn_data_num)
label_trn   =Trainnumbers.trn_label_ex(1:trn_data_num,:)';                  % Extract training labels (1*trn_data_num)
image       = 0;
[trn_after_pca,~,PCA_info]=task1_PCA(data_trn,label_trn,digit_dimension,image);
trn_af_PCA  = trn_after_pca.image;                                          % Train data after PCA

  %%% 1.2: Load the test data, and use PCA results from train data to decrease 
  %%% the dimension of test data.(In this function, test data are automatically embeded.)
file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Check and open the Testing dataset.
if exist (file_to_open_testdata,'file')
    testdata=load('Testnumbers');                                              
else
    error('check the directory of training dataset again')
end
Testnumbers = testdata.Testnumbers;
data_test        = Testnumbers.test_image_ex(:,1:test_data_num);            % Test data  (784*test_data_num)
label_test       = Testnumbers.test_label_ex(1:test_data_num,:)';           % Test label (1*test_data_num)
mean_trn     = PCA_info.mean_trn;                                           %(784*1 vec):contains mean of trn data of each pattern.
std_trn      = PCA_info.std_trn;                                            % Standard deviation of trn data of each pattern.
transformation_matrix = PCA_info.transformation_matrix;                     % Transformation matrix (digit_dimension*784):t_m*cov(trn_n')*t_m'=diagnal matrix which contains biggest NY.
data_test_normal = data_test;
std_index    = find(std_trn~=0);
for ii = 1:test_data_num                                                    % Normalize the testing data based on mean and variance of training data
    data_test_normal(std_index,ii) = (data_test(std_index,ii)-mean_trn(std_index))./std_trn(std_index);
end
test_af_PCA = transformation_matrix*data_test_normal;                       % Testing data after PCA (digit_dimension*trn_data_num)

  %%% 1.3 Extract data into classes 0-9.
trn_class_complete = cell(10,1);                                            % Each cell contains all digits belonging to one label.
in_class_mean      = zeros(digit_dimension,10);                             % 10 columns contain mean value of all data belongs to individual label.
Cov_in_class       = cell(10,1);                                            % Covariance matrix of data in each class (within class scatter)
for ii = 1:10
    class_index     = find(label_trn == ii-1);                              % Find the position of data which contains same label.
    trn_per_class   = trn_af_PCA(:,class_index);                            % Extract train data with same label.
    trn_class_complete{ii,1} = trn_per_class;                               % Store the train data per label into matrix 'trn_class_complete'.
    in_class_mean(:,ii) = mean(trn_per_class')';                            % within class mean
    Cov_in_class{ii,1}  = cov(trn_per_class');                              % within class scatter    
end    


class1 = 1;
class2 = 2;

trn_c1 = trn_class_complete{class1,1};      % Train data of class1
trn_c2 = trn_class_complete{class2,1};      % Train data of class2
mean_c1 = in_class_mean(:,class1);          % Mean of all train data of class1
mean_c2 = in_class_mean(:,class2);          % Mean of all train data of class2
trn_complete = [trn_c1,trn_c2];             % dimension*num_trn

Num_c1 = length(trn_c1);                    % Number of train data of class1
Num_c2 = length(trn_c2);                    % Number of train data of class2
M = zeros(2,Num_c1+Num_c2);                 
Kernal_c1 = zeros(Num_c1+Num_c2,Num_c1);
Kernal_c2 = zeros(Num_c1+Num_c2,Num_c2);

for col = 1:(Num_c1+Num_c2)
    distance_data_class = sum( (repmat(trn_complete(:,col),1,Num_c1)-trn_c1).^2,1 );
    Kernal_c1(col,:) = exp(-distance_data_class./(2*sigma^ 2));
    M(1,col) = 1/Num_c1*sum(Kernal_c1(col,:));
end

for col = 1:(Num_c1+Num_c2)
    distance_data_class = sum( (repmat(trn_complete(:,col),1,Num_c2)-trn_c2).^2,1 );
    Kernal_c2(col,:) = exp(-distance_data_class./(2*sigma^ 2));
    M(2,col) = 1/Num_c1*sum(Kernal_c2(col,:));
end
%
L1 = 1/Num_c1*ones(Num_c1,Num_c1);
L2 = 1/Num_c2*ones(Num_c2,Num_c2);
H = Kernal_c1*(eye(Num_c1)-L1)*Kernal_c1'+Kernal_c2*(eye(Num_c2)-L2)*Kernal_c2';
mu = 50;
H_mu = H+mu*eye(Num_c1+Num_c2);

alfa = H_mu\(M(1,:)-M(2,:))';
%
Kernal_complete = [Kernal_c1,Kernal_c2];
y_class1 = alfa'*Kernal_c1;
y_class2 = alfa'*Kernal_c2;
%
% Now calculate the threshold value y0 on the surface y
m_class1_y = mean(y_class1);        % mean of data belongs to class1 projected to y surface
m_class2_y = mean(y_class2);
y_thres = (Num_c1*m_class1_y+Num_c2*m_class2_y)/(Num_c1+Num_c2);
% Normally, given a test data Xt, calculate the its projection
% on y: yt=W_opt'*Xt. Then compare yt with y_thres. If yt>
% y_thres, then Xt belongs to class1,else, Xt belongs to class2.

  %%% Test classification results
for ii = 1:test_data_num
    distance_data_class = sum( (repmat(test_af_PCA(:,ii),1,(Num_c1+Num_c2))-trn_complete).^2,1 )';
    Kernal_test(:,ii) = exp(-distance_data_class./(2*sigma^ 2));
end
%
y_test = alfa'*Kernal_test;               % Project the test data into one-dimension space: y.
classfit = zeros(1,size(y_test,2));     % Initialization of output
for ii = 1:size(y_test,2)
    if y_test(ii)>y_thres               % If test data inside space y is bigger than y threshold.
        classfit(1,ii) = class1;        % then this test data belongs to class1.
    else                                % or, it belongs to class2.
        classfit(1,ii) = class2;
    end
end




