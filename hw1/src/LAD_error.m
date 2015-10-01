function [E]=  LAD_error(w_ml, X, Y, M, lambda)
%%
%lambda=abs(lambda); 
N= length(X); 
M=M+1; 
for n=1:N
    for m=1:M
        Phi(n,m)= X(n)^(m-1); 
    end 
end 


y0t= Phi*w_ml; 

 E= sum(abs(y0t-Y))+lambda*w_ml'*w_ml;  %%+(lambda)*(w_ml)'*w_ml)); 

end