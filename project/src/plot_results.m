%% CNN Plots

%% filter_size 2-15

filter_vec= 2:15;
acc_vec= [89.06 92.97 96.09 94.53 96.09 97.26 96.48 95.7 97.26 96.8 96.88 97.66 95.7 97.2];
t_vec=[239 260 263 275 291 316 327 364 391 440 462 523 570 619];

subplot(1,2,1)
plot(filter_vec, 100-acc_vec, 'o--')
xlabel('Filter Size')
ylabel('Test Error %')
subplot(1,2,2)
plot(filter_vec, t_vec, 'o--')
xlabel('Filter Size')
ylabel('Training Time')

%% Dropout Rate 

do_rate= linspace(.5,1, 11);

do_acc=[93.75 96.09 96.09 93.75 96.48 95.3 96.4 96.0 97.76 97.65 95.43];
do_time=[280 334 285 277 291 298 300 301 305 306 306];

subplot(1,2,1)
plot(do_rate, 100-do_acc, 'o--')
xlabel('Dropout Rate')
ylabel('Test Error %')
subplot(1,2,2)
plot(do_rate,do_time, 'o--')
xlabel('Dropout Rate')
ylabel('Training Time(s)')
%%

acc=[96.9 98.0 95.7 90.6 91.4 93.8 87.9; 96.9 96.1 97.7 90.2 90.2 93.8 86.7; 94.9 95.7 95.3 85.5 92.6 93.4 90.6; 96.9 98.0 95.7 90.6 91.4 93.8 87.9; 96.9 96.1 97.7 90.2 90.2 93.7 86.7; 94.9 95.7 95.3 85.5 92.3 93.4 90.6; 13.7 96.1 96.9 88.3 89.5 94.1 91.0; 13.7 95.7 96.5 91.8 90.6 93.4 91.4; 5 95.3 95.7 88.7 91.0 93.0 91.0; 12 95.7 96.5 89.8 93.8 93.0 90.6; 10.9 96.5 96.5 86.3 94.4 93.4 90.2; 8 95.7 96.5 89.1 91.0 93.0 91.4; 12 95.3 96.1 88.3 87.1 93.0 91.8; 5 98.4 96.9 87.5 93.0 91.0 90.2; 13 96.9 94.1 90.2 90.6 93.3 89.5; 8 94.9 96.9 84.8 91.8 94.2 92.3; 8 97.2 97.3 87.9 90.2 93.0 88.3; 8 95.4 94.1 89.8 92.2 92.2 87.5; 7 96.5 93.8 87.5 87.5 91.8 89.5];
lenet1= acc(:,1);
lenet4= acc(:,2);
lenet5= acc(:,3); 
threelayerfewer= acc(:,4);
threelayermore= acc(:, 5);
twolayer1000= [acc(:,6)];
%%
max(lenet1) 
var(lenet1)
max(lenet4)
max(lenet5)
max(threelayerfewer)
max(threelayermore)
max(twolayer1000)

%%


array=[256            500 1000 2000  4000 5000 10000 20000 30000 40000 50000 55000]; 
LENET5percent= [90.6 95.3 95.3 94.5 96.01 95.1 95.1  95.3  95.3   97.2 96.8  97.5];
%plot(array, 100-LENET5percent, '-o')
%hold on 
f = fit(array',100-LENET5percent','exp1')
plot(f,array,100-LENET5percent)
hold on 
array= [ 256 500 1000 2000 4000 5000      10000 20000 30000 40000 50000 55000] ;
LENET1per= [ 89.1 91.4 92.2 94.1 94.1 95.3 93.0 94.5 93.8 93.8 94.8  94.9];   
%plot(array, 100-LENET1per, '-o')
f = fit(array',100-LENET1per','exp1')
%semilogx(array, f(array))
%hold on 
plot(f,array,100-LENET1per)

%semilogx(array,100-LENET1per)
