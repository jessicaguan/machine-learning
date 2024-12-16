clc;
% 
% % load dataset
% data = readtable('ket_user_smote.csv');
% 
% % set features to x, set target to y
% x = data(:, 1:7);
% y = data.Ketamine_User;
% 

% % used reference to code in the link below to partition the train and test
% % data
% % https://kr.mathworks.com/help/stats/predict-class-labels-using-classification-knn-predict-block.html
% % create partition in the data 
% rng('default')
% cv = cvpartition(y, 'Holdout', 0.2); % 80% training, 20% testing
% 
% % get indices for training data and testing data
% trainIdx = training(cv);
% testIdx = test(cv);
% 
% % split the data into training data and testing data
% xTrain = x(trainIdx, :);
% yTrain = y(trainIdx);
% 
% xTest = x(testIdx, :);
% yTest = y(testIdx);
% 
% % writetable(xTest, 'xTestRFKetamine.csv');
% % writecell(yTest, 'yTestRFKetamine.csv');
% 
% % set classNames to the classes to predict
% classNames = {'N', 'Y'};
% 
% % TRAIN MODEL
% 
% % set random seed for reproducibility 
% rng('default');
% 
% % reference: https://uk.mathworks.com/help/stats/fitcensemble.html
%
% % create decision tree learner 
% t = templateTree('Reproducible',true, 'MinLeafSize', 1, 'MaxNumSplits', 6351);
% 
% % create random forest model with fitcensemble and with the optimized
% % hyperparameters, train the model
% rf_mdl_k = fitcensemble(xTrain, yTrain, 'ClassNames', classNames,...
%     'Method', 'Bag', 'NumLearningCycles',487, 'Learners', t);
% 
% 
% y_pred_train = predict(rf_mdl_k, xTrain);
% 
% fprintf('Performance for training model');
% 
% % get confusion matrix for the training set
% trainConfMat = confusionmat(yTrain, y_pred_train);
% disp('Confusion Matrix (Training Set):');
% disp(trainConfMat);
% 
% % get resubstituion loss
% resubLossValue = resubLoss(rf_mdl_k);
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
% cvMdl = crossval(rf_mdl_k, 'KFold', 5); 
% 
% % get k fold loss
% cvkfold = kfoldLoss(cvMdl, 'LossFun', 'ClassifError');
% disp('k-fold Loss:');
% disp(cvkfold);
% % Visualize the confusion matrix
% figure;
% confusionchart(yTrain, predict(rf_mdl_k, xTrain), 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
% title('Ketamine Confusion Matrix (Training)');
% 
% fprintf('\n');

  
% TESTING

xTest = readtable('xTestRFKetamine.csv');
yTest = readtable('yTestRFKetamine.csv');
yTest = table2cell(yTest);

rng('default')
y_pred_test = predict(rf_mdl_k, xTest);


fprintf('Performance for testing model');
% Confusion matrix for the testing set
testConfMat = confusionmat(yTest, y_pred_test);
disp('Confusion Matrix (Testing Set):');
disp(testConfMat);

% get resubstition loss
resubLossValue = resubLoss(rf_mdl_k);

% get accuracy
accuracy = 1 - resubLossValue;
disp(['Accuracy (Test Set): ' num2str(accuracy)]);

% get precision
precision = testConfMat(2, 2) / (testConfMat(2, 2) + testConfMat(1, 2));
disp(['Precision (Test Set): ' num2str(precision)]);

% create cross validated model
cvMdlA = crossval(rf_mdl_k, 'KFold', 5); 

% get k fold loss
cvkfold = kfoldLoss(cvMdlA, 'LossFun', 'ClassifError');
disp('k-fold Loss:');
disp(cvkfold);

% Visualize the confusion matrix
figure;
confusionchart(yTest, y_pred_test, 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
title('Ketamine Confusion Matrix (Testing Set)');

fprintf('\n');