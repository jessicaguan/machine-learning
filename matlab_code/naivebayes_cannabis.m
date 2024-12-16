clc;

% 
% % load datatset
% data = readtable('cannabis_user_smote.csv');
% 
% % set features to x, set target to y
% x = data(:, 1:7);
% y = data.Cannabis_User;
% 

% % used reference to code in the link below to partition the train and test
% % data
% % https://kr.mathworks.com/help/stats/predict-class-labels-using-classification-knn-predict-block.html
% 
% % create partition in the data
% rng('default')
% cv = cvpartition(y, 'Holdout', 0.2); % 80% training, 20% testing
% 
% % get indices for training data and testing data
% trainIdx = training(cv,1);
% testIdx = test(cv,1);
% 
% % set training data to xTrain and yTrain
% xTrain = x(trainIdx, :);
% yTrain = y(trainIdx);
% 
% % set testing data to xTest and yTest
% xTest = x(testIdx, :);
% yTest = y(testIdx);
% 
% % writetable(xTest, 'xTestNBCannabis.csv');
% % writecell(yTest, 'yTestNBCannabis.csv');
% 
% % set classNames to the classes to predict
% classNames = {'N','Y'};
% 
% % TRAINING
% 
% % reference: https://uk.mathworks.com/help/stats/fitcnb.html
%
% % create naive bayes model with normal distribution and empirical prior
% nb_mdl_c = fitcnb(xTrain, yTrain, 'ClassNames', classNames,... 
%     'DistributionNames', 'normal', 'Prior', 'empirical');
% 
% fprintf('Performance for training model with kernel distribution and empirical prior: \n');
% 
% % get confusion matrix
% trainConfMat = confusionmat(yTrain, predict(nb_mdl_c, xTrain));
% disp('Confusion Matrix (Training Set):');
% disp(trainConfMat);
% 
% % get resubstitution loss 
% resubLossValue = resubLoss(nb_mdl_c);
% 
% % get accuracy
% accuracy = 1 - resubLossValue;
% disp(['Accuracy (Training Set): ' num2str(accuracy)]);
% 
% % get precision
% precision = trainConfMat(2, 2) / (trainConfMat(2, 2) + trainConfMat(1, 2));
% disp(['Precision (Training Set): ' num2str(precision)]);
% 
% % create cross validated model
% cvMdl = crossval(nb_mdl_c);     
% 
% % get k fold loss
% cvkfold = kfoldLoss(cvMdl);
% disp('K fold loss:');
% disp(cvkfold);
% 
% figure;
% confusionchart(yTrain, predict(nb_mdl_c, xTrain), 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
% title('Cannabis Confusion Matrix (Training)');
% 
% fprintf('\n');


% TESTING
        
% load test data
 xTest = readtable('xTestNBCannabis.csv');
 yTest = readtable('yTestNBCannabis.csv');
 yTest = table2cell(yTest);

rng('default')
y_pred_test = predict(nb_mdl_c, xTest);

fprintf('Performance for testing: \n');

% get confusion matrix
testConfMat = confusionmat(yTest, y_pred_test);
disp('Confusion Matrix (Testing Set):');
disp(testConfMat);

% get resubstitution loss 
resubLossValue = resubLoss(nb_mdl_c);
        
% get accuracy
accuracy = 1 - resubLossValue;
disp(['Accuracy (Testing Set): ' num2str(accuracy)]);
        
% get precision
precision = testConfMat(2, 2) / (testConfMat(2, 2) + testConfMat(1, 2));
disp(['Precision (Testing Set): ' num2str(precision)]);

% create cross validated model
cvMdlA = crossval(nb_mdl_c);     

% get k fold loss
cvkfold = kfoldLoss(cvMdlA);
disp('K fold loss:');
disp(cvkfold);

% referenced link below for code to get figure for confusion matrix
% https://uk.mathworks.com/help/deeplearning/ug/custom-training-experiment-using-parallel-workers.html
figure;
confusionchart(yTest, predict(nb_mdl_c, xTest), 'ColumnSummary', ...
    'column-normalized', 'RowSummary', 'row-normalized');
title('Cannabis Confusion Matrix (Testing Set)');
        
fprintf('\n');
