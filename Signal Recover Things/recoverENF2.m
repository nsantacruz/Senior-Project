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

fs = 1000;
decf = 2;
framet = 2;    %seconds per frame
overt = 1;     %seconds overlap b/w frames

wlen = framet*(fs/decf);
olen = overt*(fs/decf);

decsig = decimate(filteredSig,decf);   % decimate from 1kHz to 250Hz
decsig = decsig(1:floor(length(decsig)/wlen)*wlen);  %truncate so that its divisible by wlen


f1s = zeros(1,length(decsig)/wlen);
f2s = f1s;
f3s = f2s;

%xframed = zeros(decsig/wlen,wlen);
for ii = 1:length(decsig)/wlen
    start = (ii-1)*wlen+1;
    x = decsig(start:start + olen - 1);
    %xframed(ii,:) = x;
    
    N = length(x);    % length of signal
    M = round((2*N)/5);
    P = 2;

    %create X matrix
    X = zeros(M,N);
    for jj = 1:N-M;
        X(:,jj) = x(jj:jj+M-1);
    end
    
    [U,~,~] = svd(X);
    
%     singVals = diag(S);
%     [~,singValsInds] = sort(singVals,'descend');
%     
%     Us = zeros(M,P);
%     for jj = 1:P
%         Us(:,jj) = U(:,singValsInds(jj));
%     end
    Us = U(:,1:P);
    U1 = Us(1:end-1,:);
    U2 = Us(2:end,:);
    
    Q = pinv(U1)*U2;
    
    effs = (fs/decf)*angle(eig(Q))/(2*pi);
    f1s(ii) = effs(1);
    f2s(ii) = effs(2);
    f3s(ii) = sum(effs);
    
end

figure;plot(f1s);
figure;plot(f2s);
figure;plot(f3s);






end

