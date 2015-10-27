function [w,S]=svm_test_kernel(name, num,C,bw)
disp('======Training======');
%%
% load data from csv files
data = importdata(strcat('data/data_',name,'_train.csv'));
%%
X = data(:,1:2);
Y = data(:,3);
%%
% Carry out training, primal and/or dual
eps=10^-5;
as = find_svm_weights_kernel(X, Y, C,bw);
ws = ((as .* Y)' * X)';
w=norm(ws);
indM= find(as>eps & as<C-eps);
M= length(indM); 
indS= find(as>eps & as<=C); 
S= length(indS); 

w_0=0; 
k= @(x,z) exp(-1/(2*bw^2)*norm(x-z)^2); 
 
for j=1:M 
    for i=1:S
        w_0 = w_0- as(indS(i))*Y(indS(i))*X(indM(j),:)*X(indS(i),:)';
    end 
    w_0 = w_0 + Y(indM(j));
end 
w_0= w_0/M; 

%%
% Define the predictSVM(x) function, which uses trained parameters
% assume x is given as column it is  
    


predictSVM= @(x) predictLoop(x,S,Y,X,indS,as,bw);




hold on;
% plot training results
plotDecisionBoundary(X, Y, predictSVM, [ -1, 0, 1], 'SVM Train', num(1));
%%

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

% plot validation results
plotDecisionBoundary(X, Y, predictSVM, [ -1, 0, 1], 'SVM Validate', num(2));
end 
