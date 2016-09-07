%%% Author:         Yichao Zhang
%%% Version:        2.0
%%% Date:           2015-09-28
%%% Description:
  % Back propagation is a method of training artificial neural networks
  % used in conjunction with an optimization method such as gradient
  % descent. The method calculates the gradient of a cost function with
  % respect to all the weights and bias in the network. The gradient is
  % fed to the optimization method which in turn uses it to update the
  % weights and bias, in attempt to minimize the cost function.


digit_dimension                = 100;                                       % Number of patterns want to keep
BP_parameter.epoch             = 10000;                                     % Number of iterations 
BP_parameter.num_input_neural  = digit_dimension;                           % Input layer number, which equals to the digit dimension after PCA.
BP_parameter.num_hidden_neural = 200;                                       % Hidden layer number: normally equals two times the input layer number.
BP_parameter.num_output_neural = 10;                                        % In this case equals to 10.
BP_parameter.goal              = 0.01;                                      % Performance goal (the iteration stops when average cost is below this value)
BP_parameter.lr                = 0.5;                                       % Learning rate
BP_parameter.mini_batch        = 100;

[cost_value_average_V1,BP_net_V1,test_info_V1] = BP_V1_train_cost_quadratic(digit_dimension,BP_parameter);
% This function trains the model of weighting matrix and bias vector in
% order to decrease the average cost after each BP iteration until reaching
% the threshold.
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

BP_cost_plot = 1;
accuracy_V1 = back_propagation_V1_test(test_info_V1,BP_net_V1,cost_value_average_V1,BP_cost_plot);
% This function test the model trained above, and see the digits detection
% rate.

BP_parameter.epoch             = 5000;                                     % Number of iterations 
[cost_value_average_V2,BP_net_V2,test_info_V2] = BP_V2_train_cost_entropy(digit_dimension,BP_parameter);

accuracy_V2 = back_propagation_V1_test(test_info_V2,BP_net_V2,cost_value_average_V2,BP_cost_plot);
%%
BP_parameter.epoch             = 1000;                                     % Number of iterations 
lamda = 5;
[cost_value_average_V4,BP_net_V4,test_info_V4] = BP_V4_train_L2regular(digit_dimension,BP_parameter,lamda);

accuracy_V4 = back_propagation_V1_test(test_info_V4,BP_net_V4,cost_value_average_V4,BP_cost_plot);












