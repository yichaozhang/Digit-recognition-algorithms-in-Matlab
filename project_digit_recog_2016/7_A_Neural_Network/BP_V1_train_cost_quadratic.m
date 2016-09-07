%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-09-04
%%% Description:
  % Back propagation is a method of training artificial neural networks
  % used in conjunction with an optimization method such as gradient
  % descent. The method calculates the gradient of a cost function with
  % respect to all the weights and bias in the network. The gradient is
  % fed to the optimization method which in turn uses it to update the
  % weights and bias, in attempt to minimize the cost function.
  % In this function, cost function is set as: C=0.5*||y-a||^2 (quadratic
  % cost), and initially randomly select the values of weights and bias.
%%% Input
  % digit_dimension (scalar): PCA will be used to decrease the dimension of
  %   image from 784 to the value user set. By decreasing the dimension of
  %   input, BP algorithm will run more quickly and more precisely;
  % BP_parameter (struct), which includes:
  %   epoch: Number of iterations ;
  %   num_input_neural : Input layer number, which equals to the digit dimension after PCA.
  %   num_hidden_neural: Hidden layer number: normally equals two times the input layer number.
  %   num_output_neural: In this case equals to 10;
  %   goal: Performance goal (the iteration stops when average cost is below this value)
  %   lr: learning rate  
%%% Output
  % cost_value_average: M*1 vector, Each value in this vector is the average cost of one  
  %   BP iteration. For each train data, after BP we will get a cost value. For each 
  %   iteration, we input a set of train data, and we will get a cost value in average.  
  % BP_net (struct), which contains:
  %   wei_input_hidden   (num_hidden_neural*digit_dimension matrix): input layer to hidden layer weighting; 
  %   bias_input_hidden  (num_hidden_neural*1 vector): hidden layer bias.
  %   wei_hidden_output  (num_output_neural*num_hidden_neural matrix): hidden layer to output layer weighting;
  %   bias_hidden_output (num_output_neural*1 vector): output layer bias.
  % test_info (struct), contains:
  %   test_af_PCA (digit_dimension*test_data_num matrix)
  %   label_test (1*test_data_num matrix)
  
function [cost_value_average,BP_net,test_info] = BP_V1_train_cost_quadratic(digit_dimension,BP_parameter)

global main_folder  
global BP_net_V1_str trn_data_num test_data_num 

file_to_open_BP_net = [main_folder,'49_data\',BP_net_V1_str];
if ~exist(file_to_open_BP_net,'file')                                       % Check whether the output of this function has been generated
                                                                            % If is, then just load the data file and no need to run the codes below.
%%% 1st step: Pre-processing
  %%% 1.1. Load the train data, and use PCA to decrease the dimension.
  %%% (In this function, train data are automatically embeded.)
file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data = load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
data_trn     = Trainnumbers.trn_image_ex(:,1:trn_data_num);                 % Extract training dataset (784*NX)
label_trn    = Trainnumbers.trn_label_ex(1:trn_data_num,:)';                % Extract training labels (1*NX)
image        = 0;                                                           % Do not show the images of digits after PCA
[trn_after_pca,~,PCA_info]=task1_PCA(data_trn,label_trn,digit_dimension,image);
trn_af_PCA    = trn_after_pca.image;                                        % Train data after PCA
trn_label_bin = zeros(10,length(label_trn));                                % Transfer train label in scalar to 10*1 vector. 
for ii = 1:length(label_trn)                                                % Each digit after translation has same scale as BP output.
    label_select = label_trn(:,ii);
    trn_label_bin(label_select+1,ii) = 1;
end

  %%% 1.2: Load the validation data, and use PCA results from train data to decrease 
  %%% the dimension of test data.
  %%% Recall that MNIST training dataset has 60000 samples, thus, if we
  %%% choose N train data, the data left are regarded as validation data.
  %%% (In this function, test data are automatically embeded.)
data_valid     = Trainnumbers.trn_image_ex(:,trn_data_num+1:end);           % Extract validation dataset (784*(60000-NX))
label_valid    = Trainnumbers.trn_label_ex(trn_data_num+1:end,:)';          % Extract validation labels (1*(60000-NX))
valid_data_num = length(label_valid);
mean_trn       = PCA_info.mean_trn;                                         %(784*1 vec):contains mean of trn data of each pattern.
std_trn        = PCA_info.std_trn;                                          % Standard deviation of trn data of each pattern.
transformation_matrix = PCA_info.transformation_matrix;                     % Transformation matrix (NY*784):t_m*cov(trn_n')*t_m'=diagnal matrix which contains biggest NY.
std_index      = find(std_trn~=0);
data_valid_normal = data_valid;
for ii = 1:valid_data_num                           % Normalize the validation data based on mean and variance of training data
    data_valid_normal(std_index,ii) = (data_valid(std_index,ii)-mean_trn(std_index))./std_trn(std_index);
end
valid_af_PCA    = transformation_matrix*data_valid_normal;                  % validation data after PCA (NY*TX)
valid_label_bin = zeros(10,length(label_valid));                            % Transfer validation label in scalar to 10*1 vector. 
for ii=1:length(label_valid)                                                % Each digit after translation has same scale as BP output.
    label_select = label_valid(:,ii);
    valid_label_bin(label_select+1,ii) = 1;
end

  %%% 1.3: Load the test data, and use PCA results from train data to decrease 
  %%% the dimension of test data.(In this function, test data are automatically embeded.)
file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Check and open the Testing dataset.
if exist (file_to_open_testdata,'file')
    testdata=load('Testnumbers');                                              
else
    error('check the directory of training dataset again')
end
Testnumbers      = testdata.Testnumbers;
data_test        = Testnumbers.test_image_ex(:,1:test_data_num);            % Test data  (784*test_data_num)
label_test       = Testnumbers.test_label_ex(1:test_data_num,:)';           % Test label (1*test_data_num)
data_test_normal = data_test;
for ii = 1:test_data_num                           % Normalize the testing data based on mean and variance of training data
    data_test_normal(std_index,ii) = (data_test(std_index,ii)-mean_trn(std_index))./std_trn(std_index);
end
test_af_PCA    = transformation_matrix*data_test_normal;                    % Testing data after PCA (NY*TX)
test_label_bin = zeros(10,length(label_test));                              % Transfer test label in scalar to 10*1 vector. 
for ii=1:length(label_test)                                                 % Each digit after translation has same scale as BP output.
    label_select = label_test(:,ii);
    test_label_bin(label_select+1,ii) = 1;
end
test_info.test_af_PCA = test_af_PCA;
test_info.label_test = label_test;

%%% Step2: Back propagation + gradient descent learning.
  %%% BP provides us a way of computing the gradient of the cost function
  %%% of a single training sample. Here, BP is combined with learning
  %%% algorithm, in which we compute the gradient for many training
  %%% examples and calculate the average costs. 
  %%% 2.1 Load BP parameters.
learn_rate        = BP_parameter.lr;                                        % Learning rate
num_input_neural  = BP_parameter.num_input_neural;                          % Input layer number
num_hidden_neural = BP_parameter.num_hidden_neural;                         % Hidden layer number
num_output_neural = BP_parameter.num_output_neural;                         % Output layer number
epoch             = BP_parameter.epoch;                                     % Number of iterations  
goal              = BP_parameter.goal;                                      % Performance goal (the iteration stops when average cost is below this value)

  %%% 2.2 Initializartion
data_selec_per_time = BP_parameter.mini_batch;                                                  
% Stocastic gradient descent. For each BP iteration, randomly select some data from input data as inputs of BP.
% For each data selected, we will finally calculate a cost value, and this group of data, we calculate the mean cost. 
wei_input_hidden = 2*rand(num_hidden_neural,digit_dimension)-1;             
% (data input dimension)*(number of hidden neuron) Weight of input to hidden layer.       
bias_input_hidden = 2*rand(num_hidden_neural,1)-1;                          
% (number of hidden neurons)*1  Bias of input to hidden layer.             
wei_hidden_output = 2*rand(num_output_neural,num_hidden_neural)-1;          
% (number of hidden neuron)*(number of outputs)  Weight of hidden to output layer         
bias_hidden_output = 2*rand(num_output_neural,1)-1;                         
% (number of outout neurons)*1 Bias of hidden to output layer.
cost_value_average = [];
% M*1 vector, Each value of this vector is the average cost of one BP iteration.  
% For each train data, after BP we will get a cost value. For each iteration, 
% we input a set of train data, and we will get a cost value in average.  
iter = 1;   % Initialize the value of iteration.

while 1                                                                     % Jump out of the iteration until some requirements are met.
  %%% 2.3 Input: Randomly select some train data as BP input training dataset.
rand_index     = randi(trn_data_num,[data_selec_per_time,1]);
data_input     = trn_af_PCA(:,rand_index);                                  % Randomly select some digits from input dataset.
label_input    = trn_label_bin(:,rand_index);                               % Corresponding labels
cost_value_sum = 0;                                                         % Total cost value each BP iteration
wei_input_hidden_update   = zeros(num_hidden_neural,digit_dimension);       % Initialize weights and bias update matrix
wei_hidden_output_update  = zeros(num_output_neural,num_hidden_neural); 
bias_input_hidden_update  = zeros(num_hidden_neural,1);
bias_hidden_output_update = zeros(num_output_neural,1);  
    
for ii = 1:data_selec_per_time                                              % BP iteration begins
  %%% 2.4 Feedforward: for each layer l, compute z_l = w_l*a_(l-1)+b_l
    data_input_1  = data_input(:,ii);                                       % Select an input data
    label_input_1 = label_input(:,ii);                                      % and its label.
    [hidden_activation,diff_hidden_value] = ANN_BP_feedforward(data_input_1,wei_input_hidden,bias_input_hidden);
    % hidden_activation =sigmoid(z_x_l), z_x_l=wei_input_hidden*data_input_1+bias_input_hidden
    [output_activation,diff_output_value] = ANN_BP_feedforward(hidden_activation,wei_hidden_output,bias_hidden_output);
    % output_activation =sigmoid(z_x_L), z_x_L=wei_hidden_output*hidden_activation+bias_hidden_output
    
  %%% 2.5 Compute the output error vector delta for the output layer
    delta_x_L = (output_activation-label_input_1).*diff_output_value;       % (number of output neurons)*1
    wei_hidden_output_update = wei_hidden_output_update+delta_x_L*hidden_activation';
    bias_hidden_output_update = bias_hidden_output_update+delta_x_L;
    
  %%% 2.6 Backpropagate the error: Compute the error vector dalta for the hidden layer
    delta_x_hidden = (wei_hidden_output'*delta_x_L).*diff_hidden_value;     % (number of hidden neurons)*1
    wei_input_hidden_update = wei_input_hidden_update+delta_x_hidden*data_input_1';
    bias_input_hidden_update = bias_input_hidden_update+delta_x_hidden;
    
  %%% 2.7 Calculate the cost value of this train data, and add it to the total cost for this set of train data. 
    % Cost function in this function: 1/(2n)*sum(||y_x-a_x||^2)
    cost_value_1 = norm(label_input_1-output_activation)^2;
    cost_value_sum = cost_value_sum+cost_value_1;
end
cost_value_average(iter,1) = cost_value_sum/data_selec_per_time;            % After one DP iteration, we will get an average cost value for the set of training data
  %%% 2.8 Update the weights and bias according to the gradient descent.
wei_input_hidden = wei_input_hidden-learn_rate/data_selec_per_time*wei_input_hidden_update;
wei_hidden_output = wei_hidden_output-learn_rate/data_selec_per_time*wei_hidden_output_update;
bias_input_hidden = bias_input_hidden-learn_rate/data_selec_per_time*bias_input_hidden_update;
bias_hidden_output = bias_hidden_output-learn_rate/data_selec_per_time*bias_hidden_output_update;
  %%% 2.9 Before start a new iteration, check the criteria if we can jump out of the iteration.
  %%% In this case, iterations stop if any of two criterians below are met:
  %%% We have reached the maximum number of iteration, or average cost of
  %%% each iteration reaches variable 'goal'.
if (iter == epoch) || (abs(cost_value_average(iter,1))<=goal)
    break;
end
iter = iter+1
end

% Output savings
BP_net.wei_input_hidden   = wei_input_hidden;               % (data input dimension)*(number of hidden neuron) Weight of input to hidden layer.                     
BP_net.bias_input_hidden  = bias_input_hidden;              % (number of hidden neurons)*1  Bias of input to hidden layer.             
BP_net.wei_hidden_output  = wei_hidden_output;              % (number of hidden neuron)*(number of outputs)  Weight of hidden to output layer         
BP_net.bias_hidden_output = bias_hidden_output;             % (number of outout neurons)*1 Bias of hidden to output layer.

file_to_save = [main_folder,'49_data\',BP_net_V1_str];
save(file_to_save,'BP_net','cost_value_average','test_info')

else    % Corresponding to if sentence in the beginning
    load(file_to_open_BP_net);
end

end

