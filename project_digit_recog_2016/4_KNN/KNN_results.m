%%% Author:         Yichao Zhang
%%% Version:        1.0 (2015-08-18)
%%% Version:        2.0
%%% Date:           2015-09-13


file_to_open_KNN_result = [main_folder,'49_data\',KNN_result_str];
if exist(file_to_open_KNN_result,'file')                                    % Check whether KNN_result.mat exist. If is, skip all the codes left since data file has been generated.
    load (file_to_open_KNN_result);
    accuracy_compare_gray   = KNN_result.accuracy_compare_gray;             % First column of this matrix contains values of K (1,5,9,13,17,21). First row of this matrix
    accuracy_compare_binary = KNN_result.accuracy_compare_binary;           % contains values of pattern we want to keep after PCA(80,90,100,110,120,130,140,150).
                                                                            % Data in the middle show the accuracy of KNN based on the gray/binary image. (7*9 matrix)
    accuracy_gray   = accuracy_compare_gray(2:end,2:end);
    accuracy_binary = accuracy_compare_binary(2:end,2:end);                 % Extract only the accuracy value
    figure(1),
    set(figure(1),'Position',[100 20 720 460])
    plot(reshape(accuracy_gray',[],1)),hold on,
    plot(reshape(accuracy_binary',[],1))
    xlabel('index','fontsize',12)
    ylabel('accuracy','fontsize',12)
    legend('accuracy of gray images','accuracy of binary images')
    title('KNN algorithm performance comparison between gray and binary images','fontsize',13)
    image_to_save = [main_folder,'50_figure\knn\'];
    if ~exist(image_to_save,'file')                                         % If the directory of storage does not exist, create this folder.
        mkdir(image_to_save);
    end
    saveas(gcf,fullfile(image_to_save,'knn_perform_compare'),'fig')
    % The figure which plotted above shows that gray images always performs
    % better than binary ones. This is because gray image itself contains
    % more information than binary image. For a binary image, only 0 and 1
    % can be contained. However, Differences between shades can also become
    % useful pattern.
    mean_accuracy = mean(reshape(accuracy_gray',[],1));
    std_accuracy = std(reshape(accuracy_gray',[],1));
    % When checking the mean and standard deviation of accuracy of gray
    % images, the values are 0.9367 and 0.0030. This means, varied values
    % of K and pattern do not influence the accuracy of KNN a lot. The best
    % performance is 0.9423, when K=9 and patterna_num=80.
else
    error('please check whether KNN_result.mat is generated')
end 

