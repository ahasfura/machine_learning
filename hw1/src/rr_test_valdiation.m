%% plot 1.4 

train = importdata('regress_train.txt');
test = importdata('regress_test.txt');
validate= importdata('regress_validate.txt');

X = train(1,:)';
Y = train(2,:)';
xtest= test(1,:)'; 
ytest= test(2,:)'; 
xval= validate(1,:)'; 
yval= validate(2,:)'; 
%%
lambdas = logspace(-10,0,100);

for i= 1:100 
lambda=lambdas(i); 
    M=0; 
    [w_ml0]=  ridge_regression(X, Y, M, lambda); 
    y0t= zeros(size(xval))+w_ml0; 
    SSE0V(i)= (y0t-yval)'*(y0t-yval) ; %%+lambda*(norm(w_ml0))^2); 
    subplot(1,4,1)
    plot(xval, y0t, '*')
    hold on 
    %plot(X,Y, 'o')
    plot(xval, yval, 'o')
    title('M=0')

    M=1; 
    [w_ml1]=  ridge_regression(X,Y, M, lambda); 
    y1t= w_ml1(1)+w_ml1(2)*xval;  
      SSE1V(i)= (y1t-yval)'*(y1t-yval); %%+lambda*(norm(w_ml1))^2);; 

    subplot(1,4,2)
    plot(xval, y1t, '*')
     hold on 
    %plot(X,Y, 'o')

    plot(xval, yval, 'o')
    title('M=1')


 M=2; 
    [w_ml2]=  ridge_regression(X,Y, M, lambda); 
    y2t= w_ml2(1)+w_ml2(2)*xval+ w_ml2(3)*xval.^2; 
     SSE2V(i)= (y2t-yval)'*(y2t-yval); %%+lambda*(norm(w_ml2))^2);; 

    subplot(1,4,3)
    plot(xval, y2t, '*')
     hold on 
    plot(xval, yval, 'o')
    title('M=2')


    M=3; 
    [w_ml3]=  ridge_regression(X,Y, M, lambda); 
    y3t= w_ml3(1)+w_ml3(2)*xval+ w_ml3(3)*xval.^2+w_ml3(4)*xval.^3;
     SSE3V(i)= (y3t-yval)'*(y3t-yval) ; %%+lambda*(norm(w_ml3))^2); 

    subplot(1,4,4)
    hold on 
    plot(xval, y3t, '*')
    plot(xval, yval, 'o')
    title('M=3')


     M=4; 
    [w_ml4]=  ridge_regression(X,Y, M, lambda); 
    y4t= w_ml4(1)+w_ml4(2)*xval+ w_ml4(3)*xval.^2+w_ml4(4)*xval.^3+ w_ml4(5)*xval.^4;
     SSE4V(i)= (y4t-yval)'*(y4t-yval); %%+lambda*(norm(w_ml4))^2); 


    M=5; 
    [w_ml5]=  ridge_regression(X,Y, M, lambda); 
    y5t= w_ml5(1)+w_ml5(2)*xval+ w_ml5(3)*xval.^2+w_ml5(4)*xval.^3+ w_ml5(5)*xval.^4+w_ml5(6)*xval.^5;
     SSE5V(i)= (y5t-yval)'*(y5t-yval) ; %+lambda*(norm(w_ml5))^2); 



 end   
%%
    [min0, i0]=min(SSE0V); 
    [min1, i1]=min(SSE1V); 
    [min2, i2]=min(SSE2V); 
    [min3, i3]=min(SSE3V); 
    [min4, i4]=min(SSE4V); 
    [min5, i5]=min(SSE5V); 

    l0= lambdas(i0); 
    l1= lambdas(i1); 
    l2= lambdas(i2); 
    l3= lambdas(i3); 
    l4= lambdas(i4); 
    l5= lambdas(i5); 
%%
xpred= linspace(-2.5,2); 
 y4t= w_ml4(1)+w_ml4(2)*xpred+ w_ml4(3)*xpred.^2+w_ml4(4)*xpred.^3+ w_ml4(5)*xpred.^4;
plot (xtest,ytest, 'k*')
hold on 
plot(xpred, y4t, '-')

%%
(SSE0+SSE0V)/2
(SSE1+SSE1V)/2
(SSE2+SSE2V)/2
(SSE3+SSE3V)/2


%%
for i= 1:11 
lambda=lambdas(i); 
    M=0; 
    [w_ml0]=  ridge_regression(X, Y, M, lambda); 
    y0t= zeros(size(xtest))+w_ml0; 
    SSE0(i)= .5*((y0t-ytest)'*(y0t-ytest)+lambda*(norm(w_ml0))^2); 
    subplot(1,4,1)
    plot(xtest, y0t, '*')
    hold on 
    %plot(X,Y, 'o')
    plot(xtest, ytest, 'o')
    title('M=0')

    M=1; 
    [w_ml1]=  ridge_regression(X,Y, M, lambda); 
    y1t= w_ml1(1)+w_ml1(2)*xtest;  
      SSE1(i)= .5*((y1t-ytest)'*(y1t-ytest)+lambda*(norm(w_ml1))^2);; 

    subplot(1,4,2)
    plot(xtest, y1t, '*')
     hold on 
    %plot(X,Y, 'o')

    plot(xtest, ytest, 'o')
    title('M=1')

 M=2; 
    [w_ml2]=  ridge_regression(X,Y, M, lambda); 
    y2t= w_ml2(1)+w_ml2(2)*xtest+ w_ml2(3)*xtest.^2; 
     SSE2(i)= .5*((y2t-ytest)'*(y2t-ytest)+lambda*(norm(w_ml2))^2);; 

    subplot(1,4,3)
    plot(xtest, y2t, '*')
     hold on 
    plot(xtest, ytest, 'o')
    title('M=2')

    M=3; 
    [w_ml3]=  ridge_regression(X,Y, M, lambda); 
    y3t= w_ml3(1)+w_ml3(2)*xtest+ w_ml3(3)*xtest.^2+w_ml3(4)*xtest.^3;
     SSE3(i)= .5*((y3t-ytest)'*(y3t-ytest)+lambda*(norm(w_ml3))^2);
    subplot(1,4,4)
    hold on 
    plot(xtest, y3t, '*')
    plot(xtest, ytest, 'o')
    title('M=3')


 end   
