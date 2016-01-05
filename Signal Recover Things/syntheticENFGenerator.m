clc, clear, close all;
if false
mu = 60; %Hz
sigma = 0.0133;
fs = 400;
framet = 1; % seconds
numframes = 50;
framelen = framet*fs;

enf = randn(1,numframes)*sigma + mu;   %gaussian freqs with mean mu and std sig
phi = rand(1,numframes) * 2*pi;      %random phases in U[0,2pi]
%figure; plot(f); title('efffff');
%figure;plot(enf); title('enf');

fullsignal = zeros(1,numframes*framelen);

t = linspace(0,framet,framelen);

for ii = 1:numframes

    sinwave = sin(2*pi*enf(ii)*t + phi(ii));
    
    %timedomain = timedomain + sinwave;
    starti = (ii-1)*framelen + 1;
    fullsignal(starti:starti+framelen-1) = sinwave;
end

fullsignal = filter(1, [1 -0.97],fullsignal);

noise = 20;
noisysig = awgn(fullsignal,noise);


figure;plot(fullsignal); title('full non noisy sig');
figure;plot(noisysig);title('noisy sig');

L = length(noisysig);
ft = fft(noisysig);
P2 = abs(ft/L);
P1 = P2(1:round(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);

eff = fs*(0:(L/2))/L;


save(['synthEnfN=' int2str(noise) 'T=' int2str(framet) 'FS=' int2str(fs) '.mat'],'noisysig','enf','fullsignal','fs','framet');

figure;plot(eff,P1(1:end));title('FFT');

figure;spectrogram(fullsignal);
figure;spectrogram(noisysig);
else
%% ESPRIT TESTING

noises = [10,20]; % does the algorithm need to know the noise level? because if yes we need to find a way to determine it...
enfs = cell(size(noises));

estFt = 1;
estOt = 1;
estDec = 1;
fs = 400;
wlen = floor(estFt*(fs/estDec));
olen = floor(estOt*(fs/estDec));

Ms =  floor(wlen/3);%:10:ceil(2*olen/3);
Ms = floor(wlen/3):10:ceil(2*wlen/3)
Mscores = zeros(size(Ms,2),size(noises,2));

% questions / notes
% 1) how do i implement maxalgo?
% 2) basic breakdown of each algorithm?
% 3) do the algorithms know the noise levels? if so we need to change, if
% not we need to estimate
% 4)plot correlation
% 5) plot against different M's? (which M are we using right now?)

for ii = 1:length(noises)
    n = noises(ii);
    enffile = load(['synthEnfN=' int2str(n) 'T=1FS=400.mat']);
    noisysig = enffile.noisysig;
    fullsignal = enffile.fullsignal;
    enf = enffile.enf;
    fs = enffile.fs;
    
    figure;plot(fullsignal);title('non noisy signal');
    figure;plot(noisysig);title('noisy signal');
    
%     sampEnf = zeros(1,50);    %TODO make dynamic based on predicted size of esstEnf
% 
%     enf = decimate(enf,estDec);
%     for kk = 1:length(enf)/olen
%         starti = (kk-1)*olen+1;
%         endi = starti + olen -1;
% 
%         sampEnf(kk) = enf(starti); 
%     end
    
    h = waitbar(0,['Please wait... ' int2str(ii)]);
    for jj = 1:length(Ms)
        %estEnf = maxfreqalgo(fs,estDec,estFt*fs,estOt*fs,8,60,1,noisysig);
        estEnf = esprit(fs,estDec,estFt,estOt,noisysig,Ms(jj));
        %trueEnf = maxfreqalgo(fs,estDec,estFt*fs,estOt*fs,8,60,1,fullsignal);
        %trueEnf = esprit(fs,estDec,estFt,estOt,fullsignal,Ms(jj));
        
        %figure;plot(estEnf);
%        tempTD = zeros(size(timedomain));
        
%         for kk = 1:length(estEnf)
%             startT = ((kk-1)*olen+1)/fs;
%             t = linspace(startT,startT + estOt,estFt*fs);
%             sinwave = sin(2*pi*estEnf(ii)*t);
%             
%             
%             
%             tempTD = tempTD + sinwave;
%         end
        
        
        tempCorr = corrcoef(enf,estEnf);
        Mscores(jj,ii) = tempCorr(2);
        
        waitbar(jj/length(Ms));
    end
    close(h);
%     figure;
%     plot(1:length(estEnf),estEnf,1:length(estEnf),sampEnf);
%     title('estimate vs real ENF');
%     legend('estimate','real');

    
end

figure;plot(1:length(estEnf)-1,estEnf(1:end-1),1:length(estEnf)-1,enf(1:end-1));
legend('estimate','truth');
%%

x = Ms;
figure;
plot(x,Mscores(:,1),x,Mscores(:,2)); %,x,Mscores(:,3),x,Mscores(:,4));
title('M per noisy signal');
xlabel('M');
ylabel('X Correlation Avg');
legend('Noise = -3dB','Noise = 0dB','Noise = 5dB','Noise = 10dB');





end

