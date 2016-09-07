%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-08-18
%%% Description:    This function shows the results and accuracy of KNN
  % when varying the value of K and number of patterns kept after using
  % PCA.
%%% INPUT
  % Trainnumbers(struct), which includes data and labels of number 0-10.
  % Each struct contains:
  %   trn_image_ex(784*60000 matrix):60000 is the number of data/number, and
  %   784 is the number of pixels/pattern in each image (28*28)
  %   trn_image_binary(784*60000 matrix):Transfer the gray images above into
  %   binary images. Thus, pixels inside only have 0 and 1.
  %   trn_image_binary(1*60000):Labels of all data.
  % Testnumbers(struct), which includes data and labels of number 0-10.
  % Each struct contains:
  %   test_image_ex(784*60000 matrix):60000 is the number of data/number, 
  %   and 784 is the number of pixels/pattern in each image (28*28)
  %   test_image_binary(784*60000 matrix):Transfer the gray images above 
  %   into binary images. Thus, pixels inside only have 0 and 1.
  %   test_image_binary(1*60000):Labels of all data.
%%% OUTPUT
  % KNN_result(struct), which includes:
  % accuracy_compare_gray(7*9 matrix): First column of this matrix contains
  % values of K (1,5,9,13,17,21). First row of this matrix contains values
  % of pattern we want to keep after PCA(80,90,100,110,120,130,140,150). 
  % Data in the middle show the accuracy of KNN based on the gray image.
  % accuracy_compare_binary(7*9 matrix): Data in the middle show the
  % accuracy of KNN based on the binary image. 

function KNN_perform_compare(void)
global main_folder KNN_result_str
global test_data_num

file_to_open_KNN_result = [main_folder,'49_data\',KNN_result_str];
if ~exist(file_to_open_KNN_result,'file')                                   % Check whether KNN_result.mat exist. If is, skip all the codes left since data file has been generated.
    accuracy_compare_gray   = [];
    accuracy_compare_binary = [];
    file_to_open_testdata   = [main_folder,'49_data\Testnumbers.mat'];      % Check and open the Testing dataset.
    if exist (file_to_open_testdata,'file')
        testdata=load('Testnumbers');                                              
    else
        error('check the directory of training dataset again')
    end
    Testnumbers = testdata.Testnumbers;
    data_test   = Testnumbers.test_image_ex(:,1:test_data_num);             % Testing data (gray) (784*TX)
    data_test_b = Testnumbers.test_image_binary(:,1:test_data_num);         % Testing data (binary)
    label_test  = Testnumbers.test_label_ex(1:test_data_num,:)';            % Labels corresponding (1*TX)
    K_range     = 1:4:23;                                                   % Contains some K values
    accuracy_compare_gray(2:length(K_range)+1,1)   = K_range;               % First column of output matrix contains value of K
    accuracy_compare_binary(2:length(K_range)+1,1) = K_range;
    dimension_range = 80:10:150;                                            % Set the dimension of train and test data
    accuracy_compare_gray(1,2:length(dimension_range)+1)   = dimension_range;% First row of output matrix contains value of patterns
    accuracy_compare_binary(1,2:length(dimension_range)+1) = dimension_range;

    for ii = 2:length(dimension_range)+1
        NY = dimension_range(ii-1);                                         % Set the value of pattern
        for jj = 2:length(K_range)+1  
            jj
            K = K_range(jj-1);                                              % Set the value of K in KNN
    % KNN_gray
            [~,accuracy]=KNN_MNIST(data_test,label_test,NY,K);
    % KNN_Binary
            [~,accuracy_binary]=KNN_MNIST_binary(data_test_b,label_test,NY,K);
            accuracy_compare_gray(jj,ii) = accuracy;                        % Store the value of accuracy
            accuracy_compare_binary(jj,ii) = accuracy_binary;
        end
    end
    
KNN_result.accuracy_compare_gray = accuracy_compare_gray;
KNN_result.accuracy_compare_binary = accuracy_compare_binary;
save([main_folder,'49_data\',KNN_result_str],'KNN_result')                  % Save the data file
end

end
    
    
    
    
    