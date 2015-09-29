function [w_ml]=  J_ridge_regression(X, Y, lambda)
%%
[N, M]= size(X); 
%M=M+1; 

N1s=ones(N,1); 
Phi=[N1s X]; 

A = lambda * eye(M+1,M+1) + Phi' * Phi; 
B = Phi'*Y;

w_ml = A\B; 
w_ml=w_ml/norm(w_ml); 
