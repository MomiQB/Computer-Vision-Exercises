clear
close all
clc

%% load dataset
load dataset.mat

%% plot data
u = find(labels_tr==1); % selecting the training samples belonging to class 1
figure(1), hold on
plot(data_tr(u,1), data_tr(u,2),'r.')
u = find(labels_tr==2);     % selecting the training samples belonging to class 2
plot(data_tr(u,1), data_tr(u,2),'b.')
hold off

%% stratified sampling   

rng('default');  % set the random seed for reproducibility

idx_f1 = [];
idx_f2 = [];
for nclass=1:2
    u=find(labels_tr==nclass);      % select the indeces of the labels that correspond to the current class
    idx=randperm(numel(u));     % shuffle the indeces
    idx_f1 = [idx_f1; u(idx(1:round(numel(idx)/2)))];   % half of the sample in the 1st fold
    idx_f2 = [idx_f2; u(idx(1+round(numel(idx)/2):end))];   % half of the sample in the 2nd fold
end

% labels and data for the 2 folds
labels_f1 = labels_tr(idx_f1);
labels_f2=labels_tr(idx_f2);
data_f1 = data_tr(idx_f1,:);
data_f2 = data_tr(idx_f2,:);

%% train level-1 classifiers on fold1
models = {};   

% SVM with gaussian kernel
rng('default');
models{1} = fitcsvm(data_f1, labels_f1, 'KernelFunction', 'gaussian', 'KernelScale', 5);       

% SVM with polynomial kernel
rng('default');
models{2}=fitcsvm(data_f1, labels_f1, 'KernelFunction', 'polynomial', 'KernelScale', 10);

% Decision tree
rng('default');
models{3} = fitctree(data_f1, labels_f1, 'SplitCriterion', 'gdi', 'MaxNumSplits', 20);

% Naive Bayes
rng('default');
models{4} = fitcnb(data_f1, labels_f1);

% Ensemble of decision trees
rng('default');
models{5} = fitcensemble(data_f1, labels_f1);

%% make the predictions on fold2 (to be used to train the meta-classifier)
nModels = numel(models);

% initialize matrices for predictions and scores
Predictions_f2 = zeros(size(data_f2, 1), nModels);       
Scores_f2 = zeros(size(data_f2, 1), nModels);

for n=1:nModels % iterate through classifiers
    [predictions, scores] = predict(models{n}, data_f2);
    Predictions_f2(:, n) = predictions;
    Scores_f2(:, n) = scores(:,1);
end

%% train the stacked classifier on fold2
rng('default');
% Meta-classifier trained on Scores
stackedModel = fitcensemble(Scores_f2, labels_f2, "Method", "Bag");
% Meta-classifier trained on Predictions
stackedModel_pred = fitcensemble(Predictions_f2, labels_f2, "Method", "Bag");


% Vector of accuracies
ACC = [];

Predictions_te = zeros(size(data_te, 1), nModels);      
Scores_te = zeros(size(data_te, 1), nModels);

% Make predictions and compute accuracy on the testing data
for n=1:nModels
    [predictions, scores] = predict(models{n}, data_te);
    Predictions_te(:, n) = predictions;
    Scores_te(:, n) = scores(:,1);    
    ACC(n) = numel(find(predictions==labels_te)) / numel(labels_te); % computing the accuracy of the level-1 classifiers
end

% predictions of the stacked classifier trained on Scores
pred_stacked = predict(stackedModel, Scores_te);

% predictions of the stacked classifier trained on Predictions
pred_stackedPred = predict(stackedModel_pred, Predictions_te);

% Compute accuracies of meta-classifiers
ACC(nModels+1) = numel(find(pred_stacked==labels_te)) / numel(labels_te);
ACC(nModels+2) = numel(find(pred_stackedPred==labels_te)) / numel(labels_te);


disp("Accuracies of level-1 classifiers:")
disp(ACC(1:nModels));
disp("Accuracy of meta-classifier trained on scores:")
disp(ACC(nModels+1))
disp("Accuracy of meta-classifier trained on predictions:")
disp(ACC(nModels+2))

%% train level-1 classifiers on all training data

classifiers = {};  

% SVM with gaussian kernel
rng('default');
classifiers{1} = fitcsvm(data_tr, labels_tr, 'KernelFunction', 'gaussian', 'KernelScale', 5);       % 'c' stands for classification

% SVM with polynomial kernel
rng('default');
classifiers{2}=fitcsvm(data_tr, labels_tr, 'KernelFunction', 'polynomial', 'KernelScale', 10);

% Decision tree
rng('default');
classifiers{3} = fitctree(data_tr, labels_tr, 'SplitCriterion', 'gdi', 'MaxNumSplits', 20);

% Naive Bayes
rng('default');
classifiers{4} = fitcnb(data_tr, labels_tr);

% Ensemble of decision trees
rng('default');
classifiers{5} = fitcensemble(data_tr, labels_tr);

%% Make predictions on the training data

nModels = numel(classifiers); 
Predictions = zeros(size(data_tr, 1),nModels); 
Scores = zeros(size(data_tr, 1), nModels);

for n=1:nModels
    [predictions, scores] = predict(classifiers{n}, data_tr); 
    Predictions(:,n)=predictions;
    Scores(:,n)=scores(:,1);
end 



%% Train the stacked classifier on training set 
rng('default'); 
stackedModel_tr = fitcensemble(Scores, labels_tr, "Method", "Bag"); 
stackedModel_tr_pred = fitcensemble(Predictions, labels_tr, "Method", "Bag"); 

ACC2=[];
Predictions_te = zeros(size(data_te, 1),nModels); 
Scores_te = zeros(size(data_te, 1), nModels);

for n=1:nModels
    [predictions, scores] = predict(classifiers{n}, data_te); 
    Predictions_te(:,n)=predictions;
    Scores_te(:,n)=scores(:,1);
    ACC2(n)=numel(find(predictions==labels_te))/numel(labels_te);
end 

%% Predictions and accuracy

% Predictions of the meta-classifier trained on scores 
predictions_scores=predict(stackedModel_tr, Scores_te); 
ACC2(nModels+1)=numel(find(predictions_scores==labels_te))/numel(labels_te);

% Predictions of the meta-classifier trained on predictions
predictions_pred=predict(stackedModel_tr_pred, Predictions_te); 
ACC2(nModels+2)=numel(find(predictions_pred==labels_te))/numel(labels_te);

disp ('Accuracies of level-1 classifiers without stratified sampling:')
disp(ACC2(1:n))
disp ('Accuracy of meta-classifier trained on scores without stratified sampling:')
disp(ACC2(n+1))
disp ('Accuracy of meta-classifier trained on predictions without stratified sampling:')
disp(ACC2(n+2))


