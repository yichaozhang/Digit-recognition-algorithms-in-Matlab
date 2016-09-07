
global main_folder

file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data=load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
NX = 8000;
data_trn=Trainnumbers.trn_image_binary(:,1:NX);                                 % Extract training dataset (784*NX)
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
data_test = Testnumbers.test_image_binary(:,1:TX);
label_test = Testnumbers.test_label_ex(1:TX,:)';
%%
K = 35;
mean_trn = PCA_info.mean_trn;       %(784*1 vec):contains mean of trn data of each pattern.
std_trn = PCA_info.std_trn;
std_index = find(std_trn~=0);
data_test2 = mat2gray(data_test);
data_test_normal = data_test2;
for ii = 1:TX                           % Normalize the testing data based on mean and variance of training data
    data_test_normal(std_index,ii) = (data_test2(std_index,ii)-mean_trn(std_index))./std_trn(std_index);
end


transformation_matrix = PCA_info.transformation_matrix;
test_af_PCA = transformation_matrix*data_test_normal;                       % Testing data after PCA (NY*TX)
trn_image = trn_after_pca.image;
trn_label = trn_after_pca.label;
test_label_estimate = [];
for ii = 1:length(test_af_PCA)
    norm_data = sum((trn_image-repmat(test_af_PCA(:,ii),1,length(trn_image))).^2);  % Square Eucledian distance of one test data with all train data
    [B,Index_KNN] = sort(norm_data,'ascend');                               % Sort the distances from small to big
    label_KNN_list = trn_label(Index_KNN(1:K));                             % Find K smallest distance and check the corresponding trn labels
    test_label_estimate(1,ii) = mode(label_KNN_list);                       % Find the most frequent numbers appear in these labels. And this number is the label of this data
end
label_diff = label_test-test_label_estimate;
error_rate = 1-sum(label_diff(:)==0)/length(label_diff);

%% Plot
for nn=1:16
    digit = reshape(data_test_normal(:,nn),[28,28]);
    subplot(4,4,nn),imshow(digit);
end





