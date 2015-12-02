% Howard Chen, Noah Santacruz, Jon Weinrib
% Main Function for testing feature extraction and signal recovery

clc; clear all; close all;

numTestRecs = 0;
isP = 1;
Norm = 0;
Est = 0;
name1 = 'ybintrain_all_pow.mat';
name2 = 'ybintest_all_pow.mat';
name3 = 'xtrain_all_pow.mat';
name4 = 'ytrain_all_pow.mat';
[xtrain_all_pow, ytrain_all_pow, xtest, ytest, ybintrain_all_pow, ybintest_all_pow]...
    = FeatExtract(numTestRecs,...
    isP, Norm, Est, name1, name2, name3, name4);

numTestRecs = 0;
isP = 0;
Norm = 0;
Est = 0;
name1 = 'ybintrain_all_aud.mat';
name2 = 'ybintest_all_aud.mat';
name3 = 'xtrain_all_aud.mat';
name4 = 'ytrain_all_aud.mat';
[xtrain_all_aud, ytrain_all_aud, xtest, ytest, ybintrain_all_aud, ybintest_all_aud]...
    = FeatExtract(numTestRecs,...
    isP, Norm, Est, name1, name2, name3, name4);