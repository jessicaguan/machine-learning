clc

load datatset
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
trainIdx = training(cv,1);
testIdx = test(cv,1);

% set training data to xTrain and yTrain
xTrain = x(trainIdx, :);
yTrain = y(trainIdx);

% set testing data to xTest and yTest
xTest = x(testIdx, :);
yTest = y(testIdx);

% set classNames to the classes to predict
classNames = {'N','Y'};

% distribution list to iterate through to optimize hyperparameters
distributionoptions = {'normal', 'kernel', 'mvmn'};

% priors to iterate through to optimize hyperparameters
priors = {'uniform', 'empirical'};

% reference: https://uk.mathworks.com/help/stats/fitcnb.html

% grid search for training 
for d = 1:length(distributionoptions)
    for p = 1:length(priors)

        % create naive bayes model
        MdlA = fitcnb(xTrain, yTrain, 'ClassNames', classNames,... 
            'DistributionNames', distributionoptions{d}, 'Prior', priors{p});
        
        % create cross validated model
        cvMdlA = crossval(MdlA);

        
        fprintf('Performance for model with distribution %s and prior %s:\n', ...
            distributionoptions{d}, priors{p});
        
        % get confusion matrix
        trainConfMat = confusionmat(yTrain, predict(MdlA, xTrain));
        disp('Confusion Matrix (Training Set):');
        disp(trainConfMat);

        % get resubstitution loss 
        resubLossValue = resubLoss(MdlA);
        
        % get accuracy
        accuracy = 1 - resubLossValue;
        disp(['Accuracy (Training Set): ' num2str(accuracy)]);
        
        % get precision
        precision = trainConfMat(2, 2) / (trainConfMat(2, 2) + trainConfMat(1, 2));
        disp(['Precision (Training Set): ' num2str(precision)]);

        
        % get k fold loss
        cvkfold = kfoldLoss(cvMdlA);
        disp('K fold loss:');
        disp(cvkfold);

         % visualize the confusion matrix
        figure;
        confusionchart(yTrain, predict(MdlA, xTrain), 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
        title(['Confusion Matrix (Training Set) - ' distributionoptions{d} ' - ' priors{p}]);
        
        fprintf('\n');
    end
end