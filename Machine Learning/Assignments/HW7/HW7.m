
%% loading CellDNA data 
CellDNA = readtable("ML_HW_Data_CellDNA.csv");
CellDNA = table2array(CellDNA);

%% Converting target dependent variable (last column) to binary values of either 0s or 1s for your two-class classification  
Y = CellDNA(:,14);
Y(Y>0) = 1;

%% Assigning X variables and performing standardization
X = zscore(CellDNA(:,1:13));

%% SVM Model 
SVM = fitcsvm(X,Y);
sv = SVM.SupportVectors;
theta = SVM.Beta;
theta0 = SVM.Bias;
[PredictedClasses, score] = predict(SVM, X);

%% top 3 records that have the smallest **absolute** values decision values 
[A, C] = mink(abs(score(:, 1)),3);

%% decision values “wT • X + b” for the following records: 131, 165, 892, 1057
Rec = [131,165,892,1057];
for i = 1 : numel(Rec)   
    dv(i, 1) = score(Rec(i), Y(Rec(i)) + 1);
end
disp('Target_Y, wT • X + b,  Predicted_Y')
disp([Y(Rec), dv, PredictedClasses(Rec)])

%% precision, recall, F-measure of **EACH**  of the two classes
CFM = confusionmat(Y, PredictedClasses);
confusionchart(Y, PredictedClasses);
Stats = CFM_Stats(Y, PredictedClasses);

%% ROC curve for class 0
[xpos,ypos,T,AUC0] = perfcurve(Y,score(:,1),0);
figure, plot(xpos,ypos)
xlim([-0.05 1.05]), ylim([-0.05,1.05])  % shift ROC curve from axis so easier to view 
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title(['\bf ROC for class 0 by SVM, AUC0 = ' num2str(AUC0)])

%% ROC curve for class 1
[xpos,ypos,T,AUC1]=perfcurve(Y,score(:,2),1);
figure, plot(xpos,ypos)
xlim([-0.05 1.05]), ylim([-0.05,1.05])  % shift ROC curve from axis so easier to view 
xlabel('\bf FP rate'), ylabel('\bf TP rate')
title(['\bf ROC for class 1 by SVM, AUC1 = ' num2str(AUC1)])


