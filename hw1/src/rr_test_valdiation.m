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
lambdas = linspace(0,1,10);
for i= 1:10 
lambda=lambdas(i); 
    M=0; 
    [w_ml0]=  ridge_regression(X, Y, M, lambda); 
    y0t= zeros(size(xtest))+w_ml0; 
    SSE0(i)= (y0t-ytest)'*(y0t-ytest); 
    subplot(1,4,1)
    plot(xtest, y0t, '*')
    hold on 
    %plot(X,Y, 'o')
    plot(xtest, ytest, 'o')
    title('M=0')

    M=1; 
    [w_ml1]=  ridge_regression(X,Y, M, lambda); 
    y1t= w_ml1(1)+w_ml1(2)*xtest;  
      SSE1(i)= (y1t-ytest)'*(y1t-ytest); 

    subplot(1,4,2)
    plot(xtest, y1t, '*')
     hold on 
    %plot(X,Y, 'o')

    plot(xtest, ytest, 'o')
    title('M=1')

 M=2; 
    [w_ml2]=  ridge_regression(X,Y, M, lambda); 
    y2t= w_ml2(1)+w_ml2(2)*xtest+ w_ml2(3)*xtest.^2; 
     SSE2(i)= (y2t-ytest)'*(y2t-ytest); 

    subplot(1,4,3)
    plot(xtest, y2t, '*')
     hold on 
    plot(xtest, ytest, 'o')
    title('M=2')

    M=3; 
    [w_ml3]=  ridge_regression(X,Y, M, lambda); 
    y3t= w_ml3(1)+w_ml3(2)*xtest+ w_ml3(3)*xtest.^2+w_ml3(4)*xtest.^3;
     SSE3(i)= (y3t-ytest)'*(y3t-ytest); 

    subplot(1,4,4)
    hold on 
    plot(xtest, y3t, '*')
    plot(xtest, ytest, 'o')
    title('M=3')


 end   

%%
lambdas = linspace(0,1,10);
for i= 1:10 
lambda=lambdas(i); 
    M=0; 
    [w_ml0]=  ridge_regression(X, Y, M, lambda); 
    y0t= zeros(size(xval))+w_ml0; 
    SSE0(i)= (y0t-yval)'*(y0t-yval); 
    subplot(1,4,1)
    plot(xval, y0t, '*')
    hold on 
    %plot(X,Y, 'o')
    plot(xval, yval, 'o')
    title('M=0')

    M=1; 
    [w_ml1]=  ridge_regression(X,Y, M, lambda); 
    y1t= w_ml1(1)+w_ml1(2)*xval;  
      SSE1(i)= (y1t-yval)'*(y1t-yval); 

    subplot(1,4,2)
    plot(xval, y1t, '*')
     hold on 
    %plot(X,Y, 'o')

    plot(xval, yval, 'o')
    title('M=1')

 M=2; 
    [w_ml2]=  ridge_regression(X,Y, M, lambda); 
    y2t= w_ml2(1)+w_ml2(2)*xval+ w_ml2(3)*xval.^2; 
     SSE2(i)= (y2t-yval)'*(y2t-yval); 

    subplot(1,4,3)
    plot(xval, y2t, '*')
     hold on 
    plot(xval, yval, 'o')
    title('M=2')

    M=3; 
    [w_ml3]=  ridge_regression(X,Y, M, lambda); 
    y3t= w_ml3(1)+w_ml3(2)*xval+ w_ml3(3)*xval.^2+w_ml3(4)*xval.^3;
     SSE3(i)= (y3t-yval)'*(y3t-yval); 

    subplot(1,4,4)
    hold on 
    plot(xval, y3t, '*')
    plot(xval, yval, 'o')
    title('M=3')


 end   

