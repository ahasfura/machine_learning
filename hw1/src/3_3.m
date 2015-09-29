%% plot 1.4 

X = importdata('x_train.csv');
Y = importdata('y_train.csv');

xtest = importdata('x_test.csv');
ytest = importdata('y_test.csv');

xval= importdata('x_val.csv');
yval= importdata('y_val.csv');


%%
lambdas = logspace(-5,10,20);
for i= 1:20 
lambda=lambdas(i); 
 
    
    [w_ml1]=  J_ridge_regression(X,Y, lambda); 
    y1t= (w_ml1(1)+w_ml1(2:end)'*xtest')';  
      SSE1(i)= (y1t-ytest)'*(y1t-ytest); 

     y1v= (w_ml1(1)+w_ml1(2:end)'*xval')';  
      SSE1V(i)= (y1v-yval)'*(y1v-yval); 

    
end  
    

%%
lambdas = linspace(5,6,10);
for i= 1:10 
lambda=lambdas(i); 
    M=0; 
    [w_ml0]=  ridge_regression(X, Y, M, lambda); 
    y0t= zeros(size(xval))+w_ml0; 
    SSE0V(i)= (y0t-yval)'*(y0t-yval); 
    subplot(1,4,1)
    plot(xval, y0t, '*')
    hold on 
    %plot(X,Y, 'o')
    plot(xval, yval, 'o')
    title('M=0')

    M=1; 
    [w_ml1]=  ridge_regression(X,Y, M, lambda); 
   

    plot(xval, yval, 'o')
    title('M=1')

 M=2; 
    [w_ml2]=  ridge_regression(X,Y, M, lambda); 
    y2t= w_ml2(1)+w_ml2(2)*xval+ w_ml2(3)*xval.^2; 
     SSE2V(i)= (y2t-yval)'*(y2t-yval); 

    subplot(1,4,3)
    plot(xval, y2t, '*')
     hold on 
    plot(xval, yval, 'o')
    title('M=2')

    M=3; 
    [w_ml3]=  ridge_regression(X,Y, M, lambda); 
    y3t= w_ml3(1)+w_ml3(2)*xval+ w_ml3(3)*xval.^2+w_ml3(4)*xval.^3;
     SSE3V(i)= (y3t-yval)'*(y3t-yval); 

    subplot(1,4,4)
    hold on 
    plot(xval, y3t, '*')
    plot(xval, yval, 'o')
    title('M=3')


 end   


%%
(SSE0+SSE0V)/2
(SSE1+SSE1V)/2
(SSE2+SSE2V)/2
(SSE3+SSE3V)/2

