%% plot 1.4 

data = importdata('curvefitting.txt');

X = data(1,:)';
Y = data(2,:)';
xplot= linspace(0,1); 
%%
M=0; 
[w_ml0]=  WeightVec(X,Y, M); 
[SSE0, dSSE0, diff0]=sum_of_squares_error(X,Y, w_ml0, M); 
y0= zeros(size(xplot))+w_ml0; 
subplot(1,4,1)
plot(X,Y,'o', 'MarkerSize', 10)
hold on 
plot(xplot, y0)
title('M=0')

M=1; 
[w_ml1]=  WeightVec(X,Y, M); 
y1= w_ml1(1)+w_ml1(2)*xplot; 
[SSE1, dSSE1, diff1]=sum_of_squares_error(X,Y, w_ml1, M); 

subplot(1,4,2)
plot(X,Y,'o', 'MarkerSize', 10)
hold on 
plot(xplot, y1)
title('M=1')


M=3; 
[w_ml3]=  WeightVec(X,Y, M); 
y3= w_ml3(1)+w_ml3(2)*xplot+ w_ml3(3)*xplot.^2+w_ml3(4)*xplot.^3;
subplot(1,4,3)
plot(X,Y,'o', 'MarkerSize', 10)
hold on 
plot(xplot, y3)
title('M=3')


M=9; 
[w_ml9]=  WeightVec(X,Y, M); 
y9= w_ml9(1)+w_ml9(2)*xplot+ w_ml9(3)*xplot.^2+w_ml9(4)*xplot.^3+ w_ml9(5)*xplot.^4+ w_ml9(6)*xplot.^5+ w_ml9(7)*xplot.^6+ w_ml9(8)*xplot.^7+ w_ml9(9)*xplot.^8 + w_ml9(10)*xplot.^9;
subplot(1,4,4)
plot(X,Y,'o', 'MarkerSize', 10)
hold on 
plot(xplot, y9)
title('M=9')
%%

M= 1; 
wi= [1; 1]; 
f= @(x) sum_of_squares_error(X,Y, x, M);
df= @(x) d_sum_of_squares_error(X,Y, x, M); 
alpha= .01; 
eps= 10^-6; 

[x_min,f_min,i]= grad_descent(wi, alpha, eps, f, df); 

[xf, fval, ~, o]= fminunc(f, wi);
