function [test_error, val_error] = lr_test(name, num, lambda)
disp('======Training======');
% load data from csv files

data = importdata(strcat('data/data_',name,'_train.csv'));

X = data(:,1:2);
Y = data(:,3);

% Carry out training.
f= @(w_t) LR_error( w_t, X, Y, lambda); 
[w_t, f_min, int]= grad_descent([1; 1; 1] , .001, 10^(-4), f, .1);
W= w_t(2:end); 
% Define the predictLR(x) function, which uses trained parameters
predictLR= @(x) 1/(1+exp(-W'*x+w_t(1)));


for i=1:length(X)
    Ypred(i)= sign(W'*X(i,:)'+w_t(1));
end 
test_error= .5*sum(abs(Ypred-Y'));


hold on;

% plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5, 0.5], 'LR Train', num(1));

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

for i=1:length(X)
    Ypred(i)= sign(W'*X(i,:)'+w_t(1));
end 
val_error= .5*sum(abs(Ypred-Y'));

% plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5, 0.5], 'LR Validate', num(2));
