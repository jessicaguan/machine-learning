clc

% load dataset
data = readtable('ket_user_smote.csv');

% set features to x, set target to y
x = data(:, 1:7);
y = data.Ketamine_User;

% used reference to code in the link below to partition the train and test
% data
% https://kr.mathworks.com/help/stats/predict-class-labels-using-classification-knn-predict-block.html
% create partition in the data
rng('default')
cv = cvpartition(y, 'Holdout', 0.2); % 80% training, 20% testing

% get indices for training data and testing data
trainIdx = training(cv);
testIdx = test(cv);

% set training data to xTrain and yTrain
xTrain = x(trainIdx, :);
yTrain = y(trainIdx);


% set testing data to xTest and yTest
xTest = x(testIdx, :);
yTest = y(testIdx);

% set classNames to the classes to predict
classNames = {'N', 'Y'};

% set random seed for reproducibility 
rng('default');

% reference: https://uk.mathworks.com/help/stats/fitcensemble.html

% create decision tree learner for training
t = templateTree('Reproducible',true);

% create random forest model with fitcensemble and optimize hyperparameters
MdlA = fitcensemble(xTrain, yTrain, 'ClassNames', classNames, 'Learners', t,...
    'Method', 'Bag',...
    'OptimizeHyperparameters',{'NumLearningCycles','MinLeafSize',...
     'MaxNumSplits'},'HyperparameterOptimizationOptions', ...
     struct('AcquisitionFunctionName','expected-improvement-plus'));
    
% get confusion matrix for the training set
trainConfMat = confusionmat(yTrain, predict(MdlA, xTrain));
disp('Confusion Matrix (Training Set):');
disp(trainConfMat);

% get resubstituion loss
resubLossValue = resubLoss(MdlA);

% get accuracy 
accuracy = 1 - resubLossValue;
disp(['Accuracy (Training Set): ' num2str(accuracy)]);
        
% get precision
precision = trainConfMat(2, 2) / (trainConfMat(2, 2) + trainConfMat(1, 2));
disp(['Precision (Training Set): ' num2str(precision)]);

% create cross validated model
cvMdlA = crossval(MdlA, 'KFold', 5); 

% get k fold loss
cvkfold = kfoldLoss(cvMdlA, 'LossFun', 'ClassifError');
disp('k-fold Loss:');
disp(cvkfold);

% visualize the confusion matrix
figure;
confusionchart(yTrain, predict(MdlA, xTrain), 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
title('Ketamine Confusion Matrix');
        
    
fprintf('\n');