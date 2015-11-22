% Howard Chen 
% Modified by Noah Santacruz
% Audio ENF Extraction Test
clc,clear, close all;
gridLetters = 'ABCDEFGHI';
gridFreqs = [60 50 60 50 50 50 50 50 60];
recTypes = {'Pow' 'Aud'};

%--INPUT PARAMS--
grid = 'A';                         %which grid
sampleNum = 1;                     %which sample rec of grid
harmNum = 1;                        %which harmonic of enf
type = 2;                           %power = 1 audio = 2
%--END INPUT PARAMS--

A_Aud = load([grid '_' recTypes{type} '.mat'], 'tempCell');
AA1_test_Orig = A_Aud.tempCell{sampleNum};     % get samples of first audio file
AA1_max = max(abs(AA1_test_Orig));       % find maximum absolute value
AA1_test_Orig = AA1_test_Orig/AA1_max;        % scale signal

L = length(AA1_test_Orig);
ft = fft(AA1_test_Orig);
P2 = abs(ft/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = 1000*(0:(L/2))/L;
figure;plot(f,P1);

%estimate the best frequency to bandpass around
idealMidf = harmNum*gridFreqs(gridLetters == grid);
[~,ind1] = min(abs(f-(idealMidf-3)));
[~,ind2] = min(abs(f-(idealMidf+3)));
[~,maxi] = max(P1(ind1:ind2));
midf = f(maxi+ind1)

%%
fs = 1000;                          % sample rate




decF = 1;
df = 3*decF;

AA1_test = filterENF(AA1_test_Orig,idealMidf,1,decF);
%AA1_test = AA1_test_Orig;

AA1_test_len = length(AA1_test);    % length of signal
wlen = 6000;                        % window length
hop  = 100;                         % hop size
zfac = 8;                           % zero padding factor of 4

K = sum(hamming(wlen, 'periodic'))/wlen;

[s, f, t] = stft3(AA1_test, wlen, hop, zfac, fs);
[peaks, qifftSig,sfilt,ffilt] = qifft(s,f,(idealMidf-df)*decF,(idealMidf+df)*decF, fs/decF);
figure;
plot(t,abs(qifftSig));

tdmfSig = tdmf(qifftSig,50,0.03);
figure;
hold on;
   
imagesc(t,ffilt/decF,sfilt);
set(gca,'YDir','normal')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)

plot(t,tdmfSig,'b','LineWidth',1);
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
