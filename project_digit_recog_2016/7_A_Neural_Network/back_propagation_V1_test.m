%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-09-04
%%% Description:
  % After running 'back_propagation_V1_train.m', weighting matrix and bias
  % matrix have been trained. In this case, this function inputs test dataset 
  % inside and see digit recognition accuracy.
%%% Input
  % cost_value_average: M*1 vector, Each value in this vector is the average cost of one  
  %   BP iteration. For each train data, after BP we will get a cost value. For each 
  %   iteration, we input a set of train data, and we will get a cost value in average.  
  % BP_net (struct), which contains:
  %   wei_input_hidden (num_hidden_neural*digit_dimension matrix): input layer to hidden layer weighting; 
  %   bias_input_hidden (num_hidden_neural*1 vector): hidden layer bias.
  %   wei_hidden_output (num_output_neural*num_hidden_neural matrix): hidden layer to output layer weighting;
  %   bias_hidden_output (num_output_neural*1 vector): output layer bias.
  % test_info (struct), contains:
  %   test_af_PCA (digit_dimension*test_data_num matrix)
  %   label_test (1*test_data_num matrix)
  % BP_cost_plot: If ==1, then make the epoch-cost_value plot.
%%% Output
  % accuracy (scalar): show the digit recognition accuracy. (wrongly detected digit number/ test_num)

function accuracy = back_propagation_V1_test(test_info,BP_net,cost_value_average,BP_cost_plot)

global test_data_num 

test_af_PCA        = test_info.test_af_PCA;                                 % (digit_dimension*test_data_num matrix)
label_test         = test_info.label_test;                                  % (1*test_data_num matrix)
wei_input_hidden   = BP_net.wei_input_hidden;                               % (num_hidden_neural*digit_dimension matrix): input layer to hidden layer weighting; 
bias_input_hidden  = BP_net.bias_input_hidden ;                             % (num_hidden_neural*1 vector): hidden layer bias.
wei_hidden_output  = BP_net.wei_hidden_output;                              % (num_output_neural*num_hidden_neural matrix): hidden layer to output layer weighting;
bias_hidden_output = BP_net.bias_hidden_output;                             % (num_output_neural*1 vector): output layer bias.
estimated_test_label_binary = zeros(10,test_data_num);                      % Output of each estimated label is a 10*1 vector. Thus, whole test data estimated label is a 10*num_test_data scale.
for ii = 1:test_data_num                                                    % For each test data
    data_input_1 = test_af_PCA(:,ii);                                       % Select one test data
    [hidden_activation,~] = ANN_BP_feedforward(data_input_1,wei_input_hidden,bias_input_hidden);
    % Calculate the hidden layer activation vector based on the input-to-hidden weighting and bias after training.
    [output_activation,~] = ANN_BP_feedforward(hidden_activation,wei_hidden_output,bias_hidden_output);
    % Calculate the output (activation vector) based on the hidden-to-output weighting and bias after training.      
    estimated_test_label_binary(:,ii) = output_activation;                  % Store the estimated label of that test data into 'estimated_test_label_binary'
end
sim_output_binary=compet(estimated_test_label_binary);                      % For each column of 'estimated_test_label_binary', find the position of maximum value and set it as 1, and all other data in this column as 0.
label_estimate = zeros(1,test_data_num);                                    % Transfer estimated label (10*1 format) into scalar (0-9 format) for each testing data.
for ii=1:test_data_num 
    num_index = find(sim_output_binary(:,ii) == 1);
    label_estimate(1,ii) = num_index-1;
end                 

label_diff = label_test-label_estimate;
accuracy = sum(label_diff(:)==0)/length(label_diff);                        % Number of data which are correctly recognized compared to all data

if BP_cost_plot == 1
    plot(cost_value_average)
    xlabel('average cost value for each BP iteration','fontsize',12),
    ylabel('magnitude','fontsize',12)
    title('Average cost value with the increment of iteration','fontsize',13)
end 

end

