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

acc=[96.9 98.0 95.7 90.6 91.4 93.8 87.9; 96.9 96.1 97.7 90.2 90.2 93.8 86.7; 94.9 95.7 95.3 85.5 92.6 93.4 90.6];
lenet1= acc(:,1);
lenet4= acc(:,2);
lenet5= acc(:,3); 
threelayerfewer= acc(:,4);
threelayermore= acc(:, 5);
twolayer1000= acc(:,6);

