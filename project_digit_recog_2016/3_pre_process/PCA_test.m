
global data_folder
file_to_open_trn_image = [data_folder,'train-images.idx3-ubyte'];
if exist (file_to_open_trn_image,'file')
    trn_image = loadMNISTImages(file_to_open_trn_image);
else
    error('Please check the directory of dataset')
end
file_to_open_trn_label = [data_folder,'train-labels.idx1-ubyte'];
if exist (file_to_open_trn_label,'file')
    trn_label = loadMNISTLabels(file_to_open_trn_label);
else
    error('Please check the directory of dataset')
end
