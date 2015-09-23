function [x_min,f_min]= grad_descent(xi, alpha, eps, f, df) 
% where x_i is the initial guess
%   alpha is the step size
%   eps is the convergence tolerance 
%   f is a function handle that f @(x) is the objective function
%   df is a function handle that df @(x) is some function of x that is the

%%
x_old = 10*eps+xi; % make something that will for sure not converge
x_new = xi; 

while norm(x_new-x_old)> eps % check for convergence
    x_old = x_new;
    x_new = x_old-(alpha*df(x_old));
    
end 

x_min = x_new; 
f_min = f(x_min); 
