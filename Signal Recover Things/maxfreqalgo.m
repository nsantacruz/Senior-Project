function [ enf,t,sfilt,ffilt,s ] = maxfreqalgo( fs,decf,framelen,hoplen,zfac,midf,df,sig )
%MAXFREQALGO - estimates enf from time domain sig using the max frequency
%per fram algorithm 
%fs - sampling rate of sig
%decf - decimation used in algo
%framelen - number of samples per frame
%hoplen - number of samples to hop b/w frames
%zfac - multiplier by how much to zeropad signal before fft
%midf - the estimated middle frequency of enf signal
%df - the possible divergence from midf

filteredSig = filterENF(sig,midf,1,decf);
K = sum(hamming(framelen, 'periodic'))/framelen;

[s, f, t] = stft3(filteredSig, framelen, hoplen, zfac, fs);
[peaks, qifftSig,sfilt,ffilt] = qifft(s,f,(midf-df)*decf,(midf+df)*decf, fs/decf);
tdmfSig = tdmf(qifftSig,50,0.03);

enf = tdmfSig; %qifftSig; %could also be tdmfSig

end

