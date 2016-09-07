



digit_dimension = 90;

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

  %%% Step2 Fisher classification
y_thres_complete = zeros(10,10);
for c1 = 1:10            % First class
    for c2 = 1:(c1)        % Second class
            c1_class_mean     = in_class_mean(:,c1);
            c2_class_mean     = in_class_mean(:,c2);                        % Mean of in-class data
            c1_data           = trn_class_complete{c1,1}; 
            c2_data           = trn_class_complete{c2,1};                   % Training data of two classes
            Cov_in_class_c1   = Cov_in_class{c1,1}*(size(c1_data,2)-1);
            Cov_in_class_c2   = Cov_in_class{c2,1}*(size(c2_data,2)-1);     % in-class scatter
            Cov_in_class_sum  = Cov_in_class_c1+Cov_in_class_c2;            % Sw: total within class scatter.
            Cov_between_class = (c1_class_mean-c2_class_mean)*(c1_class_mean-c2_class_mean)';   % Sw: Between-class scatter
            % The Fisher cost function is: J_F(W) = W'*Sb*W/(W'*Sw*W)
            % Gradient J_F(W) with W equals to zero, we get W_opt = inv(Sw)(m1-m2)   
            W_opt = (Cov_in_class_sum)\(c1_class_mean-c2_class_mean);
            J_F = W_opt'*Cov_between_class*W_opt/(W_opt'*Cov_in_class_sum*W_opt);
            % Project all training data on one-dimension surface y.
            y_c1 = W_opt'*c1_data;          
            y_c2 = W_opt'*c2_data; 
            % Now calculate the threshold value y0 on the surface y
            m_c1_y = mean(y_c1);    % mean of data belongs to class1 projected to y surface
            m_c2_y = mean(y_c2);
            scatter_c1_y = cov(y_c1');  % in class Scatter 
            scatter_c2_y = cov(y_c2');
            Cov_in_class_sum_y = scatter_c1_y+scatter_c2_y;
            y_thres = (size(c1_data,2)*m_c1_y+size(c2_data,2)*m_c2_y)/(size(c1_data,2)+size(c2_data,2));
            % Normally, given a test data Xt, calculate the its projection
            % on y: yt=W_opt'*Xt. Then compare yt with y_thres. If yt>
            % y_thres, then Xt belongs to c1,else, Xt belongs to c2.
            y_thres_complete(c1,c2) = y_thres;
    end
end

  %%% Step3 Test classification results
y_test = W_opt'*test_af_PCA; 
class_num = zeros(10,size(test_af_PCA,2));
for ii = 1:size(test_af_PCA,2)
    for c1 = 1:10
        for c2 = 1:(c1) 
                if y_test(1,ii) > y_thres_complete(c1,c2)
                    class_num(c1,ii) = class_num(c1,ii)+1;
                else
                    class_num(c2,ii) = class_num(c2,ii)+1;
                end
        end
    end
end
%%
sim_output=compet(class_num);                      % For each column of 'class_num', find the position of maximum value and set it as 1, and all other data in this column as 0.
label_estimate = zeros(1,test_data_num);                                    % Transfer estimated label (10*1 format) into scalar (0-9 format) for each testing data.
for ii=1:test_data_num 
    num_index = find(sim_output(:,ii) == 1);
    label_estimate(1,ii) = num_index-1;
end                 

label_diff = label_test-label_estimate;
accuracy = sum(label_diff(:)==0)/length(label_diff);                        % Number of data which are correctly recognized compared to all data

        
        
        
        



