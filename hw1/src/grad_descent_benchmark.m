%%grad_descent_benchmark

%% Trial Function : QUAD BOWL 1-D

f= @(x) (x+3).^2+5;
df= @(x) 2*(x+3);

[x_min, f_min]= grad_descent(1, .5, 10^(-6), f, df);
xtest= linspace(-5,5); 
ftest= f(xtest); 
plot(xtest, ftest)
hold on 
plot(x_min, f(x_min), 'og', 'MarkerSize', 10)

%% Trial Function: QUAD BOWL N-D

xi=[1,1];
n=length(xi); 
horzoff= [3, zeros(1,length(xi)-1)];
vertoff= [5, zeros(1, length(xi)-1)];
f= @(x) (x).^2 ; 
g=@(x) 2*diag(x);    
alpha= .5*ones(size(xi));

[x_min, f_min]= grad_descent(xi,  alpha, 10^-6,  f, g); 

grid= linspace(-5,5, 25);
[X,Y] = meshgrid(grid) ;

Z= X.^2+Y.^2;

%for i=1:100 
 %  ftest(i,:)= f(xtest(:,i)');
  
%end 
surf(X,Y,Z)
hold on 
plot3(x_min(1), x_min(2), f_min, 'o', 'MarkerSize', 15, 'MarkerFaceColor', 'r')
%% Trial Function: SINE WAVE 1-D

f= @(x) sin(x); 
g=@(x) cos(x);    


xi=[1];
[x_min1, f_min1, x_path]= grad_descent_plot(xi,  .2, 10^-6,  f, g);

xi=[1];
[x_min3, f_min3, x_path3]= grad_descent_plot(xi,  .5, 10^-6,  f, g);

xi=[1];
[x_min4, f_min4, x_path4]= grad_descent_plot(xi,  1.7, 10^-6,  f, g);


xtest= linspace(-5,5); 
ftest= f(xtest); 
plot(xtest, ftest)
hold on 
plot(x_path, f(x_path), 'og', 'MarkerSize', 10)
plot(x_path3, f(x_path3), 'or', 'MarkerSize', 10)
plot(x_path4, f(x_path4), 'ok', 'MarkerSize', 10)

legend('function', '\alpha =.2', '\alpha= .5', '\alpha=1.7')
axis([-2.5, -.5, -1, -.5])
title('F(x)= sin(x)')

%% Comparison to Fmincon 

f= @(x) 3*x.^4-8*x.^3+6*x.^2 +17; 
g=@(x) 12*x.^3-24*x.^2+12*x;    

xi=.5;
tic
[x_min1, f_min1, i1]= grad_descent(xi,  .1, 10^-6,  f, g);
toc 
tic 
[xf1, fval1,~, o1]= fminunc(f, 1.5); 
toc

xi=1.5;
tic
[x_min, f_min, i]= grad_descent(xi,  .05, 10^-6,  f, g);
toc
[xf, fval, ~, o]= fminunc(f, 1.5);

xtest= linspace(-.5,2); 
ftest= f(xtest); 
plot(xtest, ftest)
hold on 
plot(x_min, f(x_min), 'og', 'MarkerSize', 10)
plot(xf, f(xf), 'xg', 'MarkerSize', 10)

plot(x_min1, f(x_min1), 'or', 'MarkerSize', 10)
%plot(xf1, f(xf1), 'xr', 'MarkerSize', 10)

legend('F(x)', 'GD x_i= 1.5', 'Fminunc x_i= 1.5', 'GD x_i= .5', 'Fminunc x_i= .5') 

%% FMinunc


