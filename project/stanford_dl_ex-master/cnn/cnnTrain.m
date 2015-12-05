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

% Configuration
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

% Load MNIST Train
filename= [pwd '/MNIST/train-images-idx3-ubyte'];
images = loadMNISTImages(filename);
images = reshape(images,imageDim,imageDim,[]);
filename= [pwd '/MNIST/train-labels-idx1-ubyte'];
labels = loadMNISTLabels(filename);
labels(labels==0) = 10; % Remap 0 to 10

% Initialize Parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);



%%

convImages = images(:, :, 1:8); 
convLabels=labels(1:8); 
%%======================================================================
%% STEP 1: Implement convNet Objective
%  Implement the function cnnCost.m.
lambda=0; 
[cost, grad, preds] = cnnCost(theta,convImages,convLabels,numClasses,...
                                filterDim,numFilters,poolDim, lambda, 0); %pred


%%======================================================================
%% STEP 2: Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

DEBUG=true;  % set this to true to check gradient
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
    db_numFilters = 2;
    db_filterDim = 9;
    db_poolDim = 5;
    db_images = images(:,:,1:10);
    db_labels = labels(1:10);
    db_theta = cnnInitParams(imageDim,db_filterDim,db_numFilters,...
                db_poolDim,numClasses);
    
    [cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,...
                                db_filterDim,db_numFilters,db_poolDim, lambda, 0);
    

    % Check gradients
    numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
                                db_labels,numClasses,db_filterDim,...
                                db_numFilters,db_poolDim, lambda, 0), db_theta);
 
    % Use this to visually compare the gradients side by side
    %disp([numGrad grad]);
    
    %diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    %disp(diff); 
 
    assert(diff < 1e-9,...
        'Difference too large. Check your gradient computation again');
    
end;

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 3;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,...
                      numFilters,poolDim, lambda, 0),theta,images,labels,options);

 fprintf('Training Finished');                 
                  
%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training
filename= [pwd '/MNIST/t10k-images-idx3-ubyte'];

testImages = loadMNISTImages(filename);
testImages = reshape(testImages,imageDim,imageDim,[]);

filename= [pwd '/MNIST/t10k-labels-idx1-ubyte'];

testLabels = loadMNISTLabels(filename);
testLabels(testLabels==0) = 10; % Remap 0 to 10
%%
[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                filterDim,numFilters,poolDim, 0, true);

acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);
