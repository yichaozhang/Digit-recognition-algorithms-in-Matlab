


digit_dimension = 90;
lr = [];

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

Weight = 2*rand(digit_dimension,10)-1;                                         % For each label, set a weight vector with digit_dimension*1 scale.
                                                                            % Thus, for 10 labels, set matrix 'Weight' with digit_dimension*10 scale.
                                         
                                                                            

for ii = 1:trn_data_num
    for label = 1:10
        if label_trn(1,ii) == label-1
            r = zeros(10,1);
            r(label,1) = 1;
            Weight = Weight + 1/ii*bsxfun(@times,repmat(trn_af_PCA(:,ii),1,10),(r-Weight'*trn_af_PCA(:,ii))');
        end
    end
    dist = Weight'*trn_af_PCA(:,ii);
    [~,max_index] = max(dist);

end


  %% 3rd step: Testing 
d_test         = Weight'*test_af_PCA;
sim_output     = compet(d_test);                                            % For each column of 'estimated_test_label_binary', find the position of maximum value and set it as 1, and all other data in this column as 0.
label_estimate = zeros(1,test_data_num);                                    % Transfer estimated label (10*1 format) into scalar (0-9 format) for each testing data.
for ii=1:test_data_num 
    num_index = find(sim_output(:,ii) == 1);
    label_estimate(1,ii) = num_index-1;
end                 

label_diff = label_test-label_estimate;
accuracy   = sum(label_diff(:)==0)/length(label_diff);                      % Number of data which are correctly recognized compared to all data

            


%%
for label = 1:10
    for ii = 1:trn_data_num
            
        if label_trn(1,ii) == label-1
            Weight(:,label) = Weight(:,label) + 1/ii*trn_af_PCA(:,ii)*(1-Weight(:,label)'*trn_af_PCA(:,ii));
        else
            Weight(:,label) = Weight(:,label) + 1/ii*trn_af_PCA(:,ii)*(-Weight(:,label)'*trn_af_PCA(:,ii));
        end
        dist = Weight'*trn_af_PCA(:,ii);
        [~,max_index] = max(dist);
        if max_index == label-1
            break
        end
    end
end








