% Howard Chen
% Audio ENF Extraction Test

clear all; close all;

A_Aud = load('A_Aud.mat', 'tempCell');

%%
fs = 1000;                          % sample rate

AA1_test = A_Aud.tempCell{1};       % get samples of first audio file
AA1_max = max(abs(AA1_test));       % find maximum absolute value
AA1_test = AA1_test/AA1_max;        % scale signal

% Decimate
decFactor = 3;
AA1_test = decimate(AA1_test, decFactor);

% Filter

% Expected ENF frequency (Hz)
f_exp = 60;
f_tol = 3; % +/- ENF frequency (Hz)

bOrd = 3;
fn = fs/2;
f1 = (f_exp-f_tol)*decFactor; 
f2 = (f_exp+f_tol)*decFactor;
w1 = f1/fn; w2 = f2/fn;
Wp = [w1 w2];
[b,a] = butter(bOrd,Wp);
AA1_test = filter(b,a,AA1_test);


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
f = f/3;

% plot the spectrogram
figure
imagesc(t, f, s)
set(gca,'YDir','normal')
set(gca, 'FontSize', 14)
ylim([0, 100]);
xlabel('Time, s')
ylabel('Frequency, Hz')
title('Amplitude Spectrogram - Audio Recording (Grid A, 1)')
handl = colorbar;
set(handl, 'FontSize', 14)
ylabel(handl, 'Magnitude, dB')


