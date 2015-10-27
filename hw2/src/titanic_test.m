function [test_error,val_error,ws]=titanic_test(name, C)
disp('======Training======');
%%
% load data from csv files
data = importdata(strcat('../hw2_resources/data/data_',name,'_train.csv'));
X = data(:,1:11);
Y = data(:,12);

% Carry out training, primal and/or dual
eps=10^-5;
as = find_svm_weights(X, Y, C);
ws = ((as .* Y)' * X)';

indM= find(as>eps & as<C-eps);
M= length(indM); 
indS= find(as>eps & as<=C); 
S= length(indS); 

    
%%  
w_0 = 1/M * sum(Y(indM)-((as(indS).*Y(indS))'*(X(indM,:)*X(indS,:)')')');

%%
% Define the predictSVM(x) function, which uses trained parameters
predictSVM = @(x) sign(ws'*x+w_0);


for i=1:length(X)
    Ypred(i)= sign(ws'*X(i,:)'+w_0);
   
end 
test_error = .5*sum(abs(Ypred-Y'))/(length(data));



%hold on;
% plot training results
%plotDecisionBoundary(X, Y, predictSVM, [ -1, 0, 1], 'SVM Train', num(1));
%%

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('../hw2_resources/data/data_',name,'_validate.csv'));
X = validate(:,1:11);
Y = validate(:,12);
for i=1:length(X)
    Ypred(i)= sign(ws'*X(i,:)'+w_0);
   
end 
val_error = .5*sum(abs(Ypred-Y'))/(length(validate));
% plot validation results
%plotDecisionBoundary(X, Y, predictSVM, [ -1, 0, 1], 'SVM Validate', num(2));

