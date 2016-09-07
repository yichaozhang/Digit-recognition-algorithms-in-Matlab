


function [classfit] = fisher_kernel_classify(class1,class2,tst_data,sigma)

global test_data_num 
global trn_class_complete 

trn_c1 = trn_class_complete{class1,1};      % Train data of class1
trn_c2 = trn_class_complete{class2,1};      % Train data of class2
trn_complete = [trn_c1,trn_c2];             % dimension*num_trn

Num_c1 = length(trn_c1);                    % Number of train data of class1
Num_c2 = length(trn_c2);                    % Number of train data of class2
M = zeros(2,Num_c1+Num_c2);                 
Kernal_c1 = zeros(Num_c1+Num_c2,Num_c1);
Kernal_c2 = zeros(Num_c1+Num_c2,Num_c2);

for col = 1:(Num_c1+Num_c2)
    distance_data_class = sum( (repmat(trn_complete(:,col),1,Num_c1)-trn_c1).^2,1 );
    Kernal_c1(col,:) = exp(-distance_data_class./(2*sigma^ 2));
    M(1,col) = 1/Num_c1*sum(Kernal_c1(col,:));
end

for col = 1:(Num_c1+Num_c2)
    distance_data_class = sum( (repmat(trn_complete(:,col),1,Num_c2)-trn_c2).^2,1 );
    Kernal_c2(col,:) = exp(-distance_data_class./(2*sigma^ 2));
    M(2,col) = 1/Num_c1*sum(Kernal_c2(col,:));
end
%
L1 = 1/Num_c1*ones(Num_c1,Num_c1);
L2 = 1/Num_c2*ones(Num_c2,Num_c2);
H = Kernal_c1*(eye(Num_c1)-L1)*Kernal_c1'+Kernal_c2*(eye(Num_c2)-L2)*Kernal_c2';
mu = 1;
H_mu = H+mu*eye(Num_c1+Num_c2);

alfa = H_mu\(M(1,:)-M(2,:))';
%
y_class1 = alfa'*Kernal_c1;
y_class2 = alfa'*Kernal_c2;
%
% Now calculate the threshold value y0 on the surface y
m_class1_y = mean(y_class1);        % mean of data belongs to class1 projected to y surface
m_class2_y = mean(y_class2);
y_thres = (Num_c1*m_class1_y+Num_c2*m_class2_y)/(Num_c1+Num_c2);
% Normally, given a test data Xt, calculate the its projection
% on y: yt=W_opt'*Xt. Then compare yt with y_thres. If yt>
% y_thres, then Xt belongs to class1,else, Xt belongs to class2.

  %%% Test classification results
for ii = 1:test_data_num
    distance_data_class = sum( (repmat(tst_data(:,ii),1,(Num_c1+Num_c2))-trn_complete).^2,1 )';
    Kernal_test(:,ii) = exp(-distance_data_class./(2*sigma^ 2));
end
%
y_test = alfa'*Kernal_test;               % Project the test data into one-dimension space: y.
classfit = zeros(1,size(y_test,2));     % Initialization of output
for ii = 1:size(y_test,2)
    if y_test(ii)>y_thres               % If test data inside space y is bigger than y threshold.
        classfit(1,ii) = class1;        % then this test data belongs to class1.
    else                                % or, it belongs to class2.
        classfit(1,ii) = class2;
    end
end

end