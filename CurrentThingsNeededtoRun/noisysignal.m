function xnoisy = noisysignal(signal,snr,band,order)
%UNTITLED3 Summary of this function goes here
band = ((band.*2)./1000)
%   Detailed explanation goes here
xtemp = awgn(signal,snr);
[b,a] = butter(order,band,'bandpass');
xnoisy = filter(b,a,xtemp);
end
% x = audioread('Train_Grid_B_A2.wav');
% xnoisy = noisysignal(x,2,[40,100],2);