function [w_ml]=  ridge_regression(X, Y, M, lambda)
%%
N= length(X); 
M=M+1; 
for n=1:N
    for m=1:M
        Phi(n,m)= X(n)^(m-1); 
    end 
end 

A = lambda * eye(M,M) + Phi' * Phi; 
B = Phi'*Y;

w_ml = A\B; 
