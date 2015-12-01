% Howard Chen, Noah Santacruz, Jon Weinrib
% Main Function for testing feature extraction and signal recovery

clc; clear all; close all;

numTestRecs = 1;
isP = 0;
Norm = 0;
Est = 0;
name1 = 'tempxtrain.mat';
name2 = 'tempytrain.mat';
name3 = 'tempxtest.mat';
name4 = 'tempytest.mat';
[xtrain, ytrain, xtest, ytest] = TempFeatExtract(numTestRecs,...
    isP, Norm, Est, name1, name2, name3, name4);