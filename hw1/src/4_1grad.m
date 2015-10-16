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
Error=[];

lambdas= logspace(-10, 1, 100); 
lambdas=0; 

for i=1:1
    lambda= lambdas(i);
    
    f= @(w_LAD) LAD_error(w_LAD, X, Y, lambda); 
    [w, f_min, int]= grad_descent_n(1 , .001, 10^(-4), f, .1);
    W(i)= w; 
    yt= (w)+zeros(size(xval)); 
    e0i=sum(abs(y0t-yval)); %%'*(y0t-yval));
    Error(i) = e0i;
    
  
 
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
 plot(xtest,ytest, '*k')
 hold on 
 plot(xpred, ypred) 
 