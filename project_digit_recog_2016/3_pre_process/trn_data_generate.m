%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-08-15
%%% Description:
  % This function generates training dataset including training data and
  % their labels from the raw MNIST dataset.
%%% Output: 
  % Trainnumbers(struct), which includes data and labels of number 0-10.
  % Each struct contains:
  %   trn_image_ex(784*60000 matrix):60000 is the number of data/number, and
  %   784 is the number of pixels/pattern in each image (28*28)
  %   trn_image_binary(784*60000 matrix):Transfer the gray images above into
  %   binary images. Thus, pixels inside only have 0 and 1.
  %   trn_image_binary(1*60000):Labels of all data.

function trn_data_generate(void)

global data_folder main_folder

trn_number = 60000;
file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Open the raw training image dataset
if ~exist (file_to_open_trndata,'file')
    file_to_open_trn_image = [data_folder,'train-images.idx3-ubyte'];
    if exist (file_to_open_trn_image,'file')
        trn_image = loadMNISTImages(file_to_open_trn_image);
    else
        error('Please check the directory of dataset')
    end
    file_to_open_trn_label = [data_folder,'train-labels.idx1-ubyte'];       % Open the raw training label dataset
    if exist (file_to_open_trn_label,'file')
        trn_label = loadMNISTLabels(file_to_open_trn_label);
    else
        error('Please check the directory of dataset')
    end
    trn_image_ex = trn_image(:,1:trn_number);                               % Select the number of training data we need
    trn_label_ex = trn_label(1:trn_number,:);
    thresh = graythresh(trn_image_ex);                                      % Global image threshold using Otsu's method
    trn_image_binary = im2bw(trn_image_ex,thresh);                          % Convert image to binary image, based on threshold
    % The output image replaces all pixels in the input image with
    % luminance greater than thresh with the value 1 (white) and replaces all other pixels with the value 0 (black). 
    Trainnumbers.trn_image_ex = trn_image_ex;
    Trainnumbers.trn_label_ex = trn_label_ex;
    Trainnumbers.trn_image_binary = trn_image_binary;
    file_to_save_trndata = [main_folder,'49_data\'];                        % Save the new dataset
    if ~exist(file_to_save_trndata,'file')                                  % If the directory of storage does not exist, create this folder.
    	mkdir(file_to_save_trndata);
    end
    save([file_to_save_trndata,'Trainnumbers.mat'], 'Trainnumbers');
end

end

