function [E]=  LAD_error(X, Y, M, lambda, xval, yval)
%%
lambda=abs(lambda); 
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

K= length(xval); 

for k=1:K
    for m=1:M
        PhiVal(k,m)= xval(k)^(m-1); 
    end 
end 


y0t= PhiVal*w_ml; 

 E= (.5*(sum(abs(y0t-yval))+(lambda)*(w_ml)'*w_ml)); 

end