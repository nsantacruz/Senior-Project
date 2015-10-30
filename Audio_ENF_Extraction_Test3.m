% Howard Chen 
% Modified by Noah Santacruz
% Audio ENF Extraction Test
clc,clear, close all;
A_Aud = load('A_Aud.mat', 'tempCell');

%%
fs = 1000;                          % sample rate

AA1_test_Orig = A_Aud.tempCell{1};     % get samples of first audio file
AA1_max = max(abs(AA1_test_Orig));       % find maximum absolute value
AA1_test_Orig = AA1_test_Orig/AA1_max;        % scale signal

AA1_test = filterENF(AA1_test_Orig,60,1,3);
%AA1_test = AA1_test_Orig;

AA1_test_len = length(AA1_test);    % length of signal
wlen = 1000;                        % window length
hop = wlen/4;                       % hop size
pad = 2;                            % zero padding factor of 4
fft_len = 1000;                     % FFT length (hop*fs*pad*factor) 

K = sum(hamming(wlen, 'periodic'))/wlen;

[s, f, t] = stft2(AA1_test, wlen, hop, fft_len, fs);

%%

% take the amplitude of fft(x) and scale it, so not to be a
% function of the length of the window and its coherent amplification
s = abs(s)/wlen/K;

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


