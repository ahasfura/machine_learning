function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);
resDim = floor(convolvedDim / poolDim);
pooledFeatures = zeros(resDim,resDim, numFilters, numImages);



% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        for poolRow = 1:resDim
            rowStart = (poolRow - 1) * poolDim + 1;
            rowEnd = rowStart + poolDim - 1;
            for poolCol = 1:resDim
                colStart = (poolCol - 1) * poolDim + 1;
                colEnd = colStart + poolDim - 1;
                patch = convolvedFeatures(rowStart:rowEnd, colStart:colEnd, ...
                                            filterNum, imageNum);
                pooledFeatures(poolRow, poolCol, filterNum, imageNum) ...
                    = mean(patch(:));
            end
        end
    end
end
