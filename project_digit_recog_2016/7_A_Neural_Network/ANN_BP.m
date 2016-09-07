

% Back propagation
global main_folder
file_to_open_trndata = [main_folder,'49_data\Trainnumbers.mat'];            % Check and open the training dataset.
if exist (file_to_open_trndata,'file')
    data=load('Trainnumbers');                                              
else
    error('check the directory of training dataset again')
end
Trainnumbers = data.Trainnumbers;
NX          = 60000;
data_trn    =Trainnumbers.trn_image_ex(:,1:NX);                             % Extract training dataset (784*NX)
label_trn   =Trainnumbers.trn_label_ex(1:NX,:)';                            % Extract training label (1*NX)
NY          = 70;                                                          % Number of patterns want to keep
image       = 0;
[trn_after_pca,~,PCA_info]=task1_PCA(data_trn,label_trn,NY,image);

file_to_open_testdata = [main_folder,'49_data\Testnumbers.mat'];            % Check and open the Testing dataset.
if exist (file_to_open_testdata,'file')
    testdata=load('Testnumbers');                                              
else
    error('check the directory of training dataset again')
end
Testnumbers = testdata.Testnumbers;
TX = 3000;
data_test_binary = Testnumbers.test_image_binary(:,1:TX);
test_image_ex = Testnumbers.test_image_ex(:,1:TX);
label_test = Testnumbers.test_label_ex(1:TX,:)';

TX               = length(label_test);                                      % Number of Testing data
mean_trn         = PCA_info.mean_trn;                                       %(784*1 vec):contains mean of trn data of each pattern.
std_trn          = PCA_info.std_trn;                                        % Standard deviation of trn data of each pattern.
std_index        = find(std_trn~=0);
data_test_normal = test_image_ex;
for ii = 1:TX                           % Normalize the testing data based on mean and variance of training data
    data_test_normal(std_index,ii) = (test_image_ex(std_index,ii)-mean_trn(std_index))./std_trn(std_index);
end
transformation_matrix = PCA_info.transformation_matrix;                     % Transformation matrix (NY*784):t_m*cov(trn_n')*t_m'=diagnal matrix which contains biggest NY.
test_af_PCA = transformation_matrix*data_test_normal;                       % Testing data after PCA (NY*TX)
trn_af_PCA  = trn_after_pca.image;                                          % Train data after PCA
trn_label   = trn_after_pca.label;                                          % Train labels


%%
trn_label_bin = zeros(10,length(trn_label));
for ii=1:length(trn_label)
    label_select=trn_label(:,ii);
    trn_label_bin(label_select+1,ii)=1;
end


bpnet = newff(minmax(trn_af_PCA),[2*NY 10],{'logsig','logsig'},'traincgb');
%When the number of neurals of hidden layer goes smaller, the computing
%speed goes faster, at the cost of larger error result.
bpnet.trainParam.show=20; 
bpnet.trainParam.epochs=1000; %训练次数设置 
bpnet.trainParam.goal=1; %训练所要达到的精度
%We set the training goal, but it does not mean that the algorithm can
%reach it. From our testing experience, each operation will get different
%result, so we let this algorithm run 5 times, if the error can reach the
%requirement, the loop will stop.
bpnet.performFcn='sse';   %性能目标值
bpnet.trainParam.mc = 0.9; %Momentum constant
bpnet.trainParam.lr=0.1;   %学习速率      
bpnet.layers{1}.initFcn='initwb'; %网络层的初始化函数选为'initwb'，使下面的输入
                                %层的初始化语句'randnr'有效
bpnet.inputWeights{1,1}.initFcn='randnr'; %输入层权值向量初始化
bpnet.inputWeights{2,1}.initFcn='randnr'; %第一网络层到第二网络层的权值向量初始化
bpnet=init(bpnet);  %初始化网络
[bpnet] = train(bpnet,trn_af_PCA,trn_label_bin);

%%

sim_output=sim(bpnet,test_af_PCA);
sim_output_binary=compet(sim_output);
label_estimate = zeros(1,TX);
for ii=1:TX
    num_index = find(sim_output_binary(:,ii) == 1);
    label_estimate(1,ii) = num_index-1;
end                 %Matrix O shows the number this algorithm gets

label_diff = label_test-label_estimate;
accuracy = sum(label_diff(:)==0)/length(label_diff);                        % Number of data which are correctly recognized compared to all data




