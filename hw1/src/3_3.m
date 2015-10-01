%% plot 1.4 

X = importdata('x_train.csv');
Y = importdata('y_train.csv');

xtest = importdata('x_test.csv');
ytest = importdata('y_test.csv');

xval= importdata('x_val.csv');
yval= importdata('y_val.csv');


%%
lambdas = logspace(0,4,100);
for i= 1:100 
    lambda=lambdas(i); 
    
    [w_ml1]=  J_ridge_regression(X,Y, lambda); 
    
    y1t= (w_ml1(1)+w_ml1(2:end)'*xtest')';  
    SSE1(i)= (y1t-ytest)'*(y1t-ytest); %% +lambda*(norm(w_ml1))^2; 

    y1v= (w_ml1(1)+w_ml1(2:end)'*xval')';  
    SSE1V(i)= (y1v-yval)'*(y1v-yval); %%  +lambda*(norm(w_ml1))^2; 

end  
%%
[min1, i1]= min(SSE1V);
l1= lambdas(i1); 



