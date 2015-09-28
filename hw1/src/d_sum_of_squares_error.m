function [dwSSE]= d_sum_of_squares_error(X,Y, w, M)

N= length(X); 
M=M+1; 
for n=1:N
    for m=1:M
        Phi(n,m)= X(n)^(m-1); 
    end 
end 

f= @(x) (Phi*x-Y)'* (Phi*x-Y); 
g= @(x) 2*Phi'*(Phi*x-Y); 
SSE= f(w); 
dwSSE=g(w); 

end 