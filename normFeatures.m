function [ normFeatMat ] = normFeatures( featMat )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

maxCols = max(abs(featMat));
%avoid division by zero
maxCols(maxCols == 0) = 1;
normFeatMat = bsxfun(@rdivide,featMat,maxCols);


end

