
%% Set up file
  
%%%%   GLOBAL VARIABLES
clear all
close all
clc
global main_folder data_folder
global KNN_result_str 
global trn_data_num test_data_num trn_data_num_small
global BP_net_V1_str BP_net_V2_str BP_net_V4_str                                        % Back propagation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST PARAMETERS 
% This section deals with the input of all the parameters required for
% initializaion of the simulation. 

main_folder = 'E:\the Netherlands\STUDY IN NETHERLAND\Master\Delft\2015-2016\pattern_recognition\project_digit_recog\';
addpath(genpath(main_folder));                                              % Add data path
data_folder = 'E:\the Netherlands\STUDY IN NETHERLAND\Master\Delft\2015-2016\pattern_recognition\project_digit_recog\100_Data_folder\'; 
addpath(genpath(data_folder));  




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm parameters

trn_data_num  = 50000;       % Since we are implementing different learning algorithms on MNIST data. 
test_data_num = 10000;       % It is necessary to set same number of training and testing data for all algorithms.

trn_data_num_small = 2000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of data file that are created within the test or simulation.
KNN_result_str = 'KNN_result.mat';
BP_net_V1_str  = ['BP_net_V1_',num2str(trn_data_num),'_trn_',num2str(test_data_num),'_tst.mat'];

BP_net_V2_str  = ['BP_net_V2_',num2str(trn_data_num),'_trn_',num2str(test_data_num),'_tst.mat'];

BP_net_V4_str  = ['BP_net_V4_',num2str(trn_data_num),'_trn_',num2str(test_data_num),'_tst.mat'];




