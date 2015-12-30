clc, clear, close all;

mu = 60; %Hz
sigma = 0.0133;
fs = 441;
framet = 1; % seconds
wlen = floor(framet*fs);

f = normrnd(mu,sigma,1,wlen);   %gaussian freqs with mean mu and std sig
phi = rand(1,wlen) * 2*pi;      %random phases in U[0,2pi]
enf = filter(1, [1 -0.97],f);

figure; plot(f); title('efffff');
figure;plot(enf); title('enf');

sig = zeros(1,wlen);
t = linspace(0,framet,wlen);
for ii = 1:wlen

    sinwave = sin(2*pi*enf(ii)*t + phi(ii));
    if (ii == -1) 
        figure;plot(t,sinwave);title(['SINUS MINUS! ' num2str(enf(ii)) ' ' num2str(phi(ii))]); 
    end
    sig = sig + sinwave;
end




%figure;plot(sig); title('final sig');

L = length(sig);
ft = fft(sig);
P2 = abs(ft/L);
P1 = P2(1:round(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);

eff = fs*(0:(L/2))/L;

%figure;plot(eff,P1(1:end-1));title('FFT');


