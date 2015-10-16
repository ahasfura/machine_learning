function [E]=  LR_error(w_t, X, Y, lambda)
%%
N= length(X); 
w_0= w_t(1); 
w=w_t(2:end); 
for n=1:N
   NNL(n) = log(1+ exp(-Y(n)*(w_0+w'*X(n,:)'))); 
end 


 E= sum(NNL)+lambda*w'*w;  %%+(lambda)*(w_ml)'*w_ml)); 

end