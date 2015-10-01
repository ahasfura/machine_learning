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
Error_0=[];

lambdas= logspace(-10, 1, 100); 
i=1; 
for i=1:100
    lambda= lambdas(i);
    
    M=0; 
    f= @(w_LAD) LAD_error(w_LAD, X, Y, M, lambda); 
    [w, f_min, int]= grad_descent_nog(.1 , .001, 10^(-4), f, .1);
    W0(i)= w; 
    y0t= (w)+zeros(size(xval)); 
    e0i=sum(abs(y0t-yval)); %%'*(y0t-yval));
    Error_0(i) = e0i;
    
   
     M=1; 
    f= @(w_LAD) LAD_error(w_LAD, X, Y, M, lambda); 
    [w, f_min, int]= grad_descent_nog([.1; .1] , .001, 10^(-4), f, .1);
    W1(i,:)= w; 
    y1t= (w(1))+w(2)*(xval); 
    e1i=(sum(abs(y1t-yval)));%%'*(y1t-yval));
    Error_1(i) = e1i;
   
     M=2; 
    f= @(w_LAD) LAD_error(w_LAD, X, Y, M, lambda); 
    [w, f_min, int]= grad_descent_nog([.1; .1; .1] , .001, 10^(-4), f, .1);
    W2(i,:)= w; 
    y2t= (w(1))+w(2)*(xval)+ w(3)*(xval.^2); 
    e2i=sum(abs(y2t-yval)); %%'*(y2t-yval));
    Error_2(i) = e2i;
    
    M=3; 
    f= @(w_LAD) LAD_error(w_LAD, X, Y, M, lambda); 
    [w, f_min, int]= grad_descent_nog([.1; .1; .1; .1] , .001, 10^(-4), f, .1);
    W3(i,:)= w; 
    y3t= (w(1))+w(2)*(xval)+ w(3)*(xval.^2)+w(4)*(xval.^3); 
    e3i=sum(abs(y3t-yval)); %%'*(y2t-yval));
    Error_3(i) = e3i;
    
    
     M=4; 
    f= @(w_LAD) LAD_error(w_LAD, X, Y, M, lambda); 
    [w, f_min, int]= grad_descent_nog([.1; .1; .1; .1; .1] , .001, 10^(-4), f, .1);
    W4(i,:)= w; 
    y4t= (w(1))+w(2)*(xval)+ w(3)*(xval.^2)+w(4)*(xval.^3)+w(5)*(xval.^4); 
    e4i=sum(abs(y4t-yval)); %%'*(y2t-yval));
    Error_4(i) = e4i;
 end 
 [min0, i0]= min(Error_0); 
 l0= lambdas(i0); 
    
[min1, i1]= min(Error_1); 
 l1= lambdas(i1); 
 w1= W1(i1,:);    
[min2, i2]= min(Error_2); 
 l2= lambdas(i2); 
 
 [min3, i3]= min(Error_3); 
 l3= lambdas(i3); 
 
 [min4, i4]= min(Error_4); 
 l4= lambdas(i4); 
 
 
 %%
 xpred= linspace(-3,2, 100); 
 ypred= w1(1)+w1(2)*xpred; 
 plot(xval,yval, 'o')
 hold on 
 plot(xpred, ypred) 
 