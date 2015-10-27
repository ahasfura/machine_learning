function [sol]= find_svm_weights(x, y, C) 
% function to find the optimal weights for soft svm classification
% inputs
% x: data
% y: classification
% outputs
% sol: optimized weight solution

%% set starting x

X0 = zeros(length(x),1); % arbitrarily chosen starting point

%% set bounds on a

LB = zeros(length(x),1); % lower bound on all values of a is 0
UB = ones(length(x),1) * C; % upper bound on all values of a is C

%% set equation constraint

Aeq = y'; % 
beq = 0;

%% set opts

optim_ver = ver('optim');                                                                                                                                                          
optim_ver = str2double(optim_ver.Version);
if optim_ver >= 6
    opts = optimset('Algorithm', 'interior-point-convex');
else
    opts = optimset('Algorithm', 'interior-point', 'LargeScale', 'off', 'MaxIter', 2000);
end

%% set H and f where 1/2*x'*H*x + f'*x

H = diag(y) * x * x' * diag(y); % this makes a matrix where every entry is t_n * t_m * x_m * x_n (linear kernel)
f = -1 * ones(length(x),1);

%% set A and b where A*x ? b

A1 = eye(length(x));
A = [-1 * A1; A1];
b1 = ones(length(x),1);
b = [b1 * 0; b1 * C]; 

%% run quadratic programming

sol = quadprog(H, f, A, b, Aeq, beq, LB, UB, X0, opts);
