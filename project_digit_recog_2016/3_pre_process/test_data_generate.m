%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-08-15
%%% Description:
  % This function generates Testing dataset including Testing data and
  % their labels from the raw MNIST dataset.
%%% Output: 
  % Testnumbers(struct), which includes data and labels of number 0-10.
  % Each struct contains:
  %   test_image_ex(784*60000 matrix):60000 is the number of data/number, 
  %   and 784 is the number of pixels/pattern in each image (28*28)
  %   test_image_binary(784*60000 matrix):Transfer the gray images above 
  %   into binary images. Thus, pixels inside only have 0 and 1.
  %   test_image_binary(1*60000):Labels of all data.

function test_data_generate(void)

global data_folder main_folder

test_number = 10000;
file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Open the raw training image dataset
if ~exist (file_to_open_testdata,'file')
    file_to_open_trn_image = [data_folder,'t10k-images.idx3-ubyte'];
    if exist (file_to_open_trn_image,'file')
        test_image = loadMNISTImages(file_to_open_trn_image);
    else
        error('Please check the directory of dataset')
    end
    file_to_open_test_label = [data_folder,'t10k-labels.idx1-ubyte'];       % Open the raw training label dataset
    if exist (file_to_open_test_label,'file')
        test_label = loadMNISTLabels(file_to_open_test_label);
    else
        error('Please check the directory of dataset')
    end
    test_image_ex = test_image(:,1:test_number);                            % Select the number of training data we need
    test_label_ex = test_label(1:test_number,:);
    thresh = graythresh(test_image_ex);                                     % Global image threshold using Otsu's method
    test_image_binary = im2bw(test_image_ex,thresh);                        % Convert image to binary image, based on threshold
    % The output image replaces all pixels in the input image with
    % luminance greater than thresh with the value 1 (white) and replaces all other pixels with the value 0 (black). 
    Testnumbers.test_image_ex = test_image_ex;
    Testnumbers.test_label_ex = test_label_ex;
    Testnumbers.test_image_binary = test_image_binary;
    file_to_save_testdata = [main_folder,'49_data\'];                        % Save the new dataset
    if ~exist(file_to_save_testdata,'file')                                  % If the directory of storage does not exist, create this folder.
    	mkdir(file_to_save_testdata);
    end
    save([file_to_save_testdata,'Testnumbers.mat'], 'Testnumbers');
end

end