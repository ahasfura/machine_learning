%% nerural nets
name= '1'; 

data=importdata(strcat('toy_multiclass_',name, '_train.csv'));

X = data(:,1:2);
Y = data(:,3);

data=importdata(strcat('toy_multiclass_',name, '_validate.csv'));

XV = data(:,1:2);
YV = data(:,3);

best_percent= 1; 
%%
lambdas= logspace(-8, 0, 10);
nodes= linspace(2, 6, 5); 
stepsize= logspace(-2,0,5); 
for i =1:1 
    lambda= lambdas(i); 
    lambda=0; 
    for j= 2:2
        node= nodes(j); 
        node= 3; 
        for k=1:1
            step=stepsize(k); 
            step=.026; 

         [W,U, Err] = neural(X,Y, lambda, k, node, XV, YV);
       
         
    [m,~]= size(X); 
    Xbias= [ones(m,1) X];  
    [m,~]= size(XV); 
    XVbias= [ones(m,1) XV];  
    f= @(x)  1./(1+exp(-x));
    a= (U*f(W*Xbias')); 
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
misclass2=find(Ypred2~=YV); misclass=find(Ypred~=Y); 
num_misclass2= length(misclass2); 
percent_misclass2= num_misclass2/length(YV)
    if best_percent > percent_misclass2
        best_percent= percent_misclass2;
        ind=[ i, j, k]; 
    end 
        end 
    end 
end


%%
figure(1)
subplot(1,2,1)
  scatter(X(:,1),X(:,2), 5, Y)
   subplot(1,2,2)  
  scatter(X(:,1),X(:,2), 5, Ypred)

%%
  subplot(1,2,1)
  scatter(XV(:,1),XV(:,2), 5, YV)
   subplot(1,2,2)  
  scatter(XV(:,1),XV(:,2), 5, Ypred2)

  
  %%
  
  name= 'test'; 

data=importdata(strcat('mnist_',name, '.csv'));

X = data(:,1:end-1);
Y = data(:,end);



%%
[W,U] = neural2(X,Y);
%%
f= @(x)  1./(1+exp(-x));

a= (U*f(W*X')); 
Pred=[];
Ypred=[];
suma= exp(a(1,:))+ exp(a(2,:))+exp(a(3,:))+exp(a(4,:))+ exp(a(5,:))+exp(a(6,:)); 
    for k= 1:6 
        Pred(k, :) = exp(a(k,:))./suma; 
    end 
Pred=Pred'; 
for i = 1:length(Y)
    [~, maxval]= max(Pred(i, :)); 
    Ypred(i)= maxval; 
end 

 Ypred=Ypred';
misclass=find(Ypred~=Y); 
num_misclass= length(misclass); 
percent_misclass= num_misclass/length(Y) 

  
  