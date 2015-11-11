%run_remy

name= '2'; 

data=importdata(strcat('toy_multiclass_',name, '_train.csv'));

X = data(:,1:2);
Y = data(:,3);

data=importdata(strcat('toy_multiclass_',name, '_validate.csv'));

XV = data(:,1:2);
YV = data(:,3);


[W,U, y]= neuralremy1(-2, -2, 4, 4);
%%

 f= @(x)  1./(1+exp(-x));
 
  [m,~]= size(X); 
    Xbias= [ones(m,1) X];  
    [m,~]= size(XV); 
    XVbias= [ones(m,1) XV];
    a= (U*f(W*Xbias')); 
    
    %%
    a2= (U*f(W*XVbias')); 
    Pred2=[]; Pred=[];
    Ypred=[]; Ypred2=[];

    
    suma= exp(a(1,:))+ exp(a(2,:))+exp(a(3,:));     suma2= exp(a2(1,:))+ exp(a2(2,:))+exp(a2(3,:)); 

    for k= 1:3 
        Pred(k, :) = exp(a(k,:))./suma; 
        Pred2(k, :) = exp(a2(k,:))./suma2; 
    end 
    Pred=Pred';   Pred2=Pred2'; 
    for n = 1:length(YV)
       [~, maxval]= max(Pred2(n, :)); 
     Ypred2(n)= maxval; 
    end 
    for n = 1:length(Y)
       [~, maxval]= max(Pred(n, :)); 
     Ypred(n)= maxval; 
    end 
Ypred=Ypred'; Ypred2=Ypred2';

%%
suptitle('Validation Data')
  subplot(1,2,1)
  scatter(XV(:,1),XV(:,2), 5, YV)
  title('Classification')
   subplot(1,2,2)  
  scatter(XV(:,1),XV(:,2), 5, Ypred2)
  title('Prediction')