function [x_min,f_min,i]= grad_descent_nog(xi, alpha0, eps, f, h) 
% where x_i is the initial guess
%   alpha is the step size
%   eps is the convergence tolerance 
%   f is a function handle that f @(x) is the objective function
%   df is a function handle that df @(x) is some function of x that is the

%%
x_old = 10*eps+xi; % make something that will for sure not converge
x_new = xi; 
i=0;
alpha=alpha0; 
while norm(x_new-x_old)> eps % check for convergence
    for j= 1:length(x_old); 
        x_plus= x_new; x_minus= x_new;     
        x_plus(j)= x_new(j)+.5*h;    x_minus(j)= x_new(j)-.5*h ;
        df(j) = (f(x_plus) - f(x_minus)) / h;

    end
    
   
    x_old = x_new;
    x_new = x_old-(alpha*df');
    i=i+1; 
    alpha=alpha0/sqrt(i+1); 
    if i>10000 
        break
        i
    end 
end 

x_min = x_new; 
f_min = f(x_min); 
