% Howard Chen, Noah Santacruz, Jon Weinrib
% Main Function for testing feature extraction and signal recovery

clc; clear all; close all;

numTestRecs = 2;
isP = 1;
Norm = 0;
Est = 0;
name1 = 'ybintrain_2off_pow.mat';
name2 = 'ybintest_2off_pow.mat';
name3 = 'xtrain_2off_pow.mat';
name4 = 'ytrain_2off_pow.mat';
name5 = 'xtest_2off_pow.mat';
name6 = 'ytest_2off_pow.mat';
name7 = 'xtltrain_2off_pow.mat';
name8 = 'xtltest_2off_pow.mat';
[xtrain_all_pow, ytrain_all_pow, xtest, ytest, ybintrain_all_pow, ybintest_all_pow,xtltrain,xtltest]...
    = FeatExtract(numTestRecs,...
    isP, Norm, Est, name1, name2, name3, name4,name5,name6,name7,name8);

% numTestRecs = 0;
% isP = 0;
% Norm = 0;
% Est = 0;
% name1 = 'ybintrain_aud.mat';
% name2 = 'ybintest_aud.mat';
% name3 = 'xtrain_aud.mat';
% name4 = 'ytrain_aud.mat';
% name5 = 'ytrain_aud.mat';
% [xtrain_aud, ytrain_aud, xtest_aud, ytest_aud, ybintrain_aud, ybintest_aud]...
%     = FeatExtract(numTestRecs,...
%     isP, Norm, Est, name1, name2, name3, name4);