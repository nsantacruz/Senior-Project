% Howard Chen, Noah Santacruz, Jon Weinrib
% Main Function for testing feature extraction and signal recovery

clc; clear all; close all;

numTestRecs = 0;
isP = 0;
Norm = 0;
Est = 0;
gridletters = 'ABCDEFGHI'
numrecgridss = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11];
for i = 1:length(gridletters)
temp = ['_',gridletters(i),'_aud.mat']
name1 = ['ybintrain',temp];
name2 = ['ybintest',temp];
name3 = ['xtrain',temp];
name4 = ['ytrain',temp];
name5 = ['xtest',temp];
name6 = ['ytest',temp];
name7 = ['xtltrain',temp];
name8 = ['xtltest',temp];
numrecgrid = numrecgridss(2*i - (~isP);
[xtrain_c_aud, ytrain_c_aud, xtest, ytest, ybintrain_c_aud, ybintest_c_aud,xtltrain_c_aud,xtltest_c_aud]...
    = FeatExtract(gridletters(i),numrecgrid,numTestRecs,...
    isP, Norm, Est, name1, name2, name3, name4,name5,name6,name7,name8);
end

