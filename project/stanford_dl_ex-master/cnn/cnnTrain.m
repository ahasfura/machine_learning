%% Convolution Neural Network Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.
imageDim = 28;

filename= [pwd '/MNIST/train-images-idx3-ubyte'];
images = loadMNISTImages(filename);
images = reshape(images,imageDim,imageDim,[]);
filename= [pwd '/MNIST/train-labels-idx1-ubyte'];
labels = loadMNISTLabels(filename);
labels(labels==0) = 10; % Remap 0 to 10

filename= [pwd '/MNIST/t10k-images-idx3-ubyte'];

testImages = loadMNISTImages(filename);
testImages = reshape(testImages,imageDim,imageDim,[]);

filename= [pwd '/MNIST/t10k-labels-idx1-ubyte'];

testLabels = loadMNISTLabels(filename);
testLabels(testLabels==0) = 10; % Remap 0 to 10


for run=11:20
epochs=3; 
lambda=0; 
% Configuration
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)



options.filterDim= filterDim; 
options.poolDim= poolDim; 
options.numFilters=numFilters;
% Load MNIST Train

% Initialize Parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);



%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = epochs;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;


tic
opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,...
                      numFilters,poolDim, lambda, 0),theta,images,labels,options);

 fprintf('Training Finished');                 
ttrain= toc;                   
%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

%%
tic
[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                filterDim,numFilters,poolDim, 0, true);
ttest= toc; 
acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);


 filename=[pwd '/stanford_dl_ex-master/results/epoch3run' num2str(run)  ]
    save(filename, 'acc', 'ttest', 'ttrain', 'options' )
 CNNacc(run)=acc; 
end 
%