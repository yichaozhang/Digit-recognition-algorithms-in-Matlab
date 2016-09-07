%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-09-04
%%% Description:
  % This function calculate activation vector of hidden layer and/or output
  % layer based on the sigmoid function.
  

function [feedforward_output,diff_feedforward_output] = ANN_BP_feedforward(data_input,weight,bias)

[feedforward_output,diff_feedforward_output] = sigmoid(weight*data_input+bias);

end


% x (N*1 vec): N is the number of inputs/digits
function [sig,diff_sig] = sigmoid(x)

sig = 1./(exp(-x)+1);
diff_sig = exp(-x)./((exp(-x)+1).^2);

end
