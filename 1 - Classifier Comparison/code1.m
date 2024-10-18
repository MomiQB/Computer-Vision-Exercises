clear
close all
clc

% Create a variable to store the results
results = [];
nDatasets = 4;


for n=1:nDatasets   % iterate through datasets
    switch n
        case 1, load dataset1.mat
        case 2, load dataset2.mat
        case 3, load dataset3.mat
        case 4, load dataset4.mat
        otherwise
    end
    
    % Arrays to store the accuracies of each model
    linearResults = [];
    rbfResults = [];
    knnResults = [];
    treeResults = [];

    for i=1:5   % repeat cross validation 5 times
        indexTrain = [];
        indexTest = [];
        % Randomly pick the indeces for training and validation samples
        for nclass=1:2
            currentClass = find(labels==nclass); % list of position of elements of the current class
            idx = randperm(numel(currentClass));  
            indexTrain = [indexTrain; currentClass(idx(1:round(numel(idx)/2)))]; 
            indexTest = [indexTest; currentClass(idx(1+round(numel(idx)/2):end))];
        end
        
        % Divide the data into training and validation sets
        labelsTrain = labels(indexTrain);
        labelsTest = labels(indexTest);
        dataTrain = data(indexTrain, :);
        dataTest = data(indexTest, :);
        
        % Training classifiers
        % training on training data, testing on testing data
        SVM_LIN = fitcsvm(dataTrain, labelsTrain, 'KernelFunction','linear', 'KernelScale',1); % LINEAR SVM 

        SVM_RBF = fitcsvm(dataTrain, labelsTrain, 'KernelFunction','gaussian', 'KernelScale',0.1); % GAUSSIAN SVM 

        KNN = fitcknn(dataTrain, labelsTrain, 'Distance','euclidean','NumNeighbors',10); % K-NEAREST NEIGHBOURS
                                                                                     
        TREE = fitctree(dataTrain, labelsTrain,'SplitCriterion','gdi','MaxNumSplits',15); % DECISION TREE
                                                                                      
        % Make predictions on unseen data (here, testing data)
        pred_SVM_LIN = predict(SVM_LIN, dataTest);
        pred_SVM_RBF = predict(SVM_RBF, dataTest);
        pred_KNN = predict(KNN, dataTest);
        pred_TREE = predict(TREE, dataTest);
        
        % Calculate accuracies
        acc1_SVM_LIN = numel(find(pred_SVM_LIN==labelsTest))/numel(labelsTest); % 'find' gives us the indices of the elements that satisfy the condition inside the parenthesis
        acc1_SVM_RBF = numel(find(pred_SVM_RBF==labelsTest))/numel(labelsTest);
        acc1_KNN = numel(find(pred_KNN==labelsTest))/numel(labelsTest);
        acc1_TREE = numel(find(pred_TREE==labelsTest))/numel(labelsTest);

        % Reverse the role of train and test
        % training on testing data, testing on training data
        SVM_LIN = fitcsvm(dataTest, labelsTest, 'KernelFunction','linear', 'KernelScale',1);
        SVM_RBF = fitcsvm(dataTest, labelsTest, 'KernelFunction','gaussian', 'KernelScale',0.2);
        KNN = fitcknn(dataTest, labelsTest, 'Distance','euclidean','NumNeighbors',10);
        TREE = fitctree(dataTest, labelsTest,'SplitCriterion','gdi','MaxNumSplits',15);
        
        % Making predictions on unseen data (here, training data)
        pred_SVM_LIN = predict(SVM_LIN, dataTrain);
        pred_SVM_RBF = predict(SVM_RBF, dataTrain);
        pred_KNN = predict(KNN, dataTrain);
        pred_TREE = predict(TREE, dataTrain);

        % Calculating accuracies
        acc2_SVM_LIN = numel(find(pred_SVM_LIN==labelsTrain))/numel(labelsTrain); % 'find' gives us the indices of the elements that satisfy the condition inside the parenthesis
        acc2_SVM_RBF = numel(find(pred_SVM_RBF==labelsTrain))/numel(labelsTrain);
        acc2_KNN = numel(find(pred_KNN==labelsTrain))/numel(labelsTrain);
        acc2_TREE = numel(find(pred_TREE==labelsTrain))/numel(labelsTrain);
        
        % Calculating final accuracies and storing them in our arrays
        acc_SVM_LIN = (acc1_SVM_LIN+acc2_SVM_LIN)/2; 
        acc_SVM_RBF = (acc1_SVM_RBF+acc2_SVM_RBF)/2;
        acc_KNN = (acc1_KNN+acc2_KNN)/2;
        acc_TREE = (acc1_TREE+acc2_TREE)/2;
        
        linearResults(i,1) = acc_SVM_LIN;
        rbfResults(i,1) = acc_SVM_RBF;
        knnResults(i,1) = acc_KNN;
        treeResults(i,1) = acc_TREE;
    end

    % Averaging the accuracies of the 5 2-fold cv and saving the results
    results(n,1) = mean(linearResults);
    results(n,2) = mean(rbfResults);
    results(n,3) = mean(knnResults);
    results(n,4) = mean(treeResults);
  
end

% Display the accuracy of each algorithm on each dataset
disp(results)

% Compute the ranking for each dataset

rankings = [];

% iterate through the results' rows
for dataset=1:nDatasets

    row = results(dataset, :);

    % Sort the vector in descending order and get the indices of the sorted elements
    [sorted_values, sorted_indices] = sort(row, 'descend');
    
    % Assign ranks
    current_rank = 1;
    for pos = 1:length(row)
        if pos > 1 && sorted_values(pos) == sorted_values(pos-1)
            % If the current value is the same as the previous one, assign an
            % average rank to both
            rankings(dataset,sorted_indices(pos)) = (current_rank + (pos-1))/2;
            rankings(dataset,sorted_indices(pos-1)) = (current_rank + (pos-1))/2;
            current_rank = current_rank + 1;
        else
            % Otherwise, assign the current rank
            rankings(dataset, sorted_indices(pos)) = current_rank;
            current_rank = current_rank + 1;
        end
    end

end


% Compute the average ranks of each algorithm
averageRanks = mean(rankings,1);
disp(averageRanks)

% CD Value
k=4; 
q_alpha= 2.291;        
CD = q_alpha*sqrt((k*(k+1))/(6*nDatasets));

% Plot the Critical Difference Diagram
figure;
plot(averageRanks, [1,2,3,4], '.', 'MarkerSize', 20)
title("Critical Difference Diagram")
xlim([0 5])
ylim([-0.5 5])
xlabel("Rankings")
ylabel("Algorithms")
yticks([1 2 3 4])
yticklabels(["LIN", "RBF", "KNN", "TREE"])

for i = 1:nDatasets
    line([averageRanks(1,i)-CD/2,averageRanks(1,i)+CD/2],[i,i], 'color', 'blue')
end