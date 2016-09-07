%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-09-13
%%% Description:
  % Suppose two classes of observations have means mu_0 and mu_1 and
  % covariances V_0 and V_1. Then the linear combination of features w'*x 
  % will have means w'*mu_i and variance w'*V_i*w for i=0,1. Fisher defined
  % the separation between these two distributions to be the ratio of the
  % variance between the classes to the variance within the classes:
  % J_F = W_opt'*Cov_between_class*W_opt/(W_opt'*Cov_in_class_sum*W_opt);
  % This measure is, in some sense, a measure of the signal-to-noise ratio
  % for the class labelling. It can be shown that the maximum separation
  % occurs when W_opt proportional to inv(Cov_in_class_sum)*(mu_1-mu_0).
  % Adterwards, W_opt is used to project the train data in two labels into 
  % one-dimensional plane: y. y_thres is set to distinguish training data
  % projecting on y plane.
%%% INPUT
  % class1 (scalar): shows the value of first label (0-9)
  % class2 (scalar): shows the value of second label (0-9)
  % tst_data (dimension*num_test_data): test data inputs.
%%% OUTPUT
  % classfit (1*num_test_data matrix): shows the classification reusults of
  % each testing data (compare class1 and class2, choose the one which is 
  % more similar to each test data)

function classfit = fisher_classify(class1,class2,tst_data)

global in_class_mean trn_class_complete Cov_in_class

c1_class_mean         = in_class_mean(:,class1);
c2_class_mean         = in_class_mean(:,class2);                            % Mean of in-class data
class1_data           = trn_class_complete{class1,1}; 
class2_data           = trn_class_complete{class2,1};                       % Training data of two classes
Cov_in_class_class1   = Cov_in_class{class1,1}*(size(class1_data,2)-1);
Cov_in_class_class2   = Cov_in_class{class2,1}*(size(class2_data,2)-1);     % in-class scatter
Cov_in_class_sum      = Cov_in_class_class1+Cov_in_class_class2;            % Sw: total within class scatter.
Cov_between_class     = (c1_class_mean-c2_class_mean)*(c1_class_mean-c2_class_mean)';   % Sw: Between-class scatter
% The Fisher cost function is: J_F(W) = W'*Sb*W/(W'*Sw*W)
% Gradient J_F(W) with W equals to zero, we get W_opt = inv(Sw)(m1-m2)   
W_opt = (Cov_in_class_sum)\(c1_class_mean-c2_class_mean);
%J_F = W_opt'*Cov_between_class*W_opt/(W_opt'*Cov_in_class_sum*W_opt);
% Project all training data on one-dimension surface y.
y_class1 = W_opt'*class1_data;          
y_class2 = W_opt'*class2_data; 
% Now calculate the threshold value y0 on the surface y
m_class1_y = mean(y_class1);        % mean of data belongs to class1 projected to y surface
m_class2_y = mean(y_class2);
y_thres = (size(class1_data,2)*m_class1_y+size(class2_data,2)*m_class2_y)/(size(class1_data,2)+size(class2_data,2));
% Normally, given a test data Xt, calculate the its projection
% on y: yt=W_opt'*Xt. Then compare yt with y_thres. If yt>
% y_thres, then Xt belongs to class1,else, Xt belongs to class2.

  %%% Test classification results
y_test = W_opt'*tst_data;               % Project the test data into one-dimension space: y.
classfit = zeros(1,size(y_test,2));     % Initialization of output
for ii = 1:size(y_test,2)
    if y_test(ii)>y_thres               % If test data inside space y is bigger than y threshold.
        classfit(1,ii) = class1;        % then this test data belongs to class1.
    else                                % or, it belongs to class2.
        classfit(1,ii) = class2;
    end
end


end









