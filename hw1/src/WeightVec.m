function [w_ml]=  WeightVec(X,Y, M)
%%
N= length(X); 
M=M+1; 
for n=1:N
    for m=1:M
        Phi(n,m)= X(n)^(m-1); 
    end 
end 

A= Phi'*Phi; 
B= Phi'*Y; 
w_ml = A\B; 
