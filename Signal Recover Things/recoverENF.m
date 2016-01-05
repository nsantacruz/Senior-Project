% Howard Chen 
% Modified by Noah Santacruz
% Audio ENF Extraction Test
function [ tdmfSig,qifftSig ] = recoverENF(grid,recType,sampleNum)
%grid = gridLetter of grid of interest (e.g. 'A')
%recType = either 'Pow' or 'Aud'
%sampleNum = for given grid, which recording sample do you want
%RETURNS
%tdmfSig - signal after stft -> qifft -> tdmf
%qifftSig - signal after stft -> qifft

datasetsRoot = '../matlabDatasets';
gridLetters = 'ABCDEFGHI';
gridFreqs = [60 50 60 50 50 50 50 50 60];

shouldPlot = true;

%--INPUT PARAMS--
%--END INPUT PARAMS--


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

%%
fs = 1000;                          % sample rate
decF = 3;
df = 1.9*decF;



framelen = 3000;                        % window length
hoplen  = 100;                         % hop size
zfac = 8;                           % zero padding factor of 4

[enf,t,sfilt,ffilt,s] = maxfreqalgo(fs,decF,framelen,hoplen,zfac,midf,df,origSig);


if shouldPlot
figure;
plot(t,abs(enf));


figure;
hold on;
   
imagesc(t,ffilt/decF,sfilt);
set(gca,'YDir','normal')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)

plot(t,enf,'b','LineWidth',1);
%plot(t,qifftSig,'r');
%colormap(hot);

legend('Output');
xlabel('Sample #');
ylabel('Frequency (Hz)');
hold off;

%[s1,f1,t1] = spectrogram(AA1_test,wlen,4096,nfft,fs,'yaxis');

%axis([1 t(end) midf-1.2*df midf+1.2*df]);
%title(['Estimated ENF Grid ' grid ' ' recTypes{type} ' ' int2str(sampleNum)]);
title('Recovered ENF Signal');
end
%%

% take the amplitude of fft(x) and scale it, so not to be a
% function of the length of the window and its coherent amplification
s = abs(s)/wlen/K;
fft_len = zfac*wlen;
% correction of the DC & Nyquist component
if rem(fft_len, 2)                     % odd nfft excludes Nyquist point
    st(2:end, :) = s(2:end, :).*2;
else                                % even nfft includes Nyquist point
    s(2:end-1, :) = s(2:end-1, :).*2;
end

% convert amplitude spectrum to dB (min = -120 dB)
s = 20*log10(s + 1e-6);

% plot the spectrogram
if shouldPlot
figure
imagesc(t, f, s)
set(gca,'YDir','normal')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
ylim([0, 300]);
xlabel('Time, s')
ylabel('Frequency, Hz')
title('Amplitude spectrogram of the signal')
handl = colorbar;
set(handl, 'FontName', 'Times New Roman', 'FontSize', 14)
ylabel(handl, 'Magnitude, dB')
end
end