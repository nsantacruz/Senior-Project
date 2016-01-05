function [ enf ] = recoverENF2( grid,recType,sampleNum )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

enf = 1;

datasetsRoot = '../matlabDatasets';
gridLetters = 'ABCDEFGHI';
gridFreqs = [60 50 60 50 50 50 50 50 60];

shouldPlot = true;

gridCellLoaded = load([datasetsRoot '/' grid '_' recType '.mat'], 'tempCell');
origSig = gridCellLoaded.tempCell{sampleNum};     % get samples of first audio file
sigMax = max(abs(origSig));       % find maximum absolute value
origSig = origSig/sigMax;        % scale signal

L = length(origSig);
ft = fft(origSig);
P2 = abs(ft/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = 1000*(0:(L/2))/L;

if shouldPlot
figure;plot(f,P1);
end

%estimate the best frequency to bandpass around
idealMidf = gridFreqs(gridLetters == grid);
[~,ind1] = min(abs(f-(idealMidf-3)));
[~,ind2] = min(abs(f-(idealMidf+3)));
[~,maxi] = max(P1(ind1:ind2));
midf = f(maxi+ind1);

fs = 1000;                          % sample rate
decF = 1;
df = 1.9*decF;

filteredSig = filterENF(origSig,idealMidf,1,decF);
%AA1_test = AA1_test_Orig;

%% ESPRIT ALGO

enf = esprit(fs,4,2,1,filteredSig);






end

