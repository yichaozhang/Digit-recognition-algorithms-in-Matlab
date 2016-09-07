%%% Author:         Yichao Zhang
%%% Version:        1.0
%%% Date:           2015-08-16
%%% Description:  principal component analysis ("PCA") rotates the
%%% original data to new coordinates, making the data as "flat" as
%%% possible.PCA is useful for data reduction, noise rejection,
%%% visualization and data compression among other things. 
%Input 
 %%% data_trn (784*NX):NX images. Each image contains 784 patterns(pixels)
 %%% label_trn (1*NX):Labels of all images.
 %%% NY: number of features. default=120
 %%% image=1 show the images after reconstruction, default=1
%Output:
 %%% trn_after_pca (struct), which contains:
   %   image(NY*NX): Matrix after PCA feature elimination;
   %   label(1*NX): Labels of all trn numbers.
 %%% reconstruction (struct), which contains:
   %   image(784*NX):This matrix can be used to plot the numbers on the
   %   screen and see the difference between original number images and
   %   estimated ones after PCA pattern elimination.
   %   label(1*NX): Labels of all trn numbers.
 %%% PCA_info (struct), which contains:
   %   mean_trn (784*1 vec):contains mean of trn data of each pattern.
   %   std_trn (784*1 vec):contains variance of trn data of each pattern.
   %   transformation_matrix (NY*784):t_m*cov(trn_n')*t_m'=diagnal matrix
   %   which contains biggest NY.
   %   eigenvalue_selected (NY*1 vec):NY highest eigenvalues of covariance
   %   matrix of original trn data.
   %   error_percent (scalar):Calculate the percent of information can be
   %   kept after using PCA.(eigenvalues deleted divide sum of all eigenvalues)

function [trn_after_pca,reconstruction,PCA_info]=task1_PCA(data_trn,label_trn,NY,image)


% 1st, normalization
[trn_n,mean_trn,std_trn] = zscore(data_trn');                               % Standardize, (X-MEAN(X)) ./ STD(X) so that the columns of
                                                                            % trn_n have sample mean zero and sample standard deviation one
mean_trn=mean_trn';                                                         % trn_n:784*NX   data after normalization
std_trn=std_trn';                                                           % mean_trn:784*1 mean(data_trn')
trn_n=trn_n';                                                               % std_trn: 784*1 var(data_trn')
% Now we have got the normalized training set and their labels

% 2nd Find out the covariance matrix of the normalized training set 
% and find its eigenvalues and eigenvectors
var_trn=cov(trn_n');                                                        % Covariance of normalized data
[vec_trn,e_trn]=eig(var_trn);
trn_eig_val_vec = diag(e_trn);                                              % Train eigenvalue vector (784*1),containing the eigenvalues of covariance of normalized data

% 3rd sort the eigenvalues in decending order and set the highest
% NY eigenvalues and their correspinding eigenvectors. 
e_trn_sort = sort(trn_eig_val_vec,'descend') ;
v_trn_sort = [];                                                            %784*NY, only contains eigenvectors of top NY eigenvalues
v_trn_sort_complete = zeros(784,784);                                       %784*784, instead of above, keep other columns 0
index_eigenvalue = find(trn_eig_val_vec >= e_trn_sort(NY));
v_trn_sort=vec_trn(:,index_eigenvalue);
v_trn_sort_complete(:,index_eigenvalue) = v_trn_sort;                       % This matrix is used to reconstruct the image.

% 4th recovering and approximation of the data
% v_trn_sort 784*NY; v_trn_sort_complete 784*784; trn_n 784*NX
new_trn_rot=v_trn_sort_complete'*trn_n;                                     % Rotation the data (Y =C'*X)
new_trn_rot2=v_trn_sort'*trn_n;                                             % trn_after_pca contains the first NY features with NX observations
data_trn_appro=v_trn_sort_complete*new_trn_rot;                             % Image reconstruction (784*NX)
reconstruction.image = data_trn_appro;
reconstruction.label = label_trn;
trn_after_pca.image = new_trn_rot2;
trn_after_pca.label = label_trn;

sum_all_eigenvalue=sum(e_trn_sort);                                         % Vector(784*1) contains all eigenvalues
sum_eigenv_eliminate = sum(e_trn_sort(NY:784));
%error_reconstruction= 0.5*sum_eigenv_eliminate;
error_percent=sum_eigenv_eliminate/sum_all_eigenvalue;

PCA_info.mean_trn = mean_trn;                                               % 784*1 vector, contains mean of trn data of each pattern.
PCA_info.std_trn  = std_trn;                                                % 784*1 vector, contains variance of trn data of each pattern.
PCA_info.transformation_matrix = v_trn_sort';                               % transformation matrix (NY*784)
PCA_info.eigenvalue_selected = e_trn_sort(1:NY);                            % NY highest eigenvalues of covariance matrix of original trn data.  
PCA_info.error_percent = error_percent;                                     % Calculate the percent of information can be kept after using PCA.

% 5th print some values to see whether any difference
if image==1
k=abs(floor(1000*rand(20,1)));
for nn=1:8
    digit = reshape(data_trn(:,k(nn)),[28,28]);
    subplot(4,4,2*nn-1),imshow(digit);
    digit = reshape(data_trn_appro(:,k(nn)),[28,28]);
    subplot(4,4,2*nn),imshow(digit);
end
end

end







