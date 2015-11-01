function [ peaks ] = qifft( s,f,t,f1,f2,offsetf )
%given s,f,t outputs of stft2, compute qifft
%s is spectrogram of signal
%f is vector representing fft frequency bins
%t are time bins
%f1 and f2 is band of interest
%ALGO: choice max bin freq BETA in b/w f1 and f2
%ALPHA is bin before, LAMBDA is after
%p = (.5*(ALPHA-LAMBDA))/(ALPHA-2*BETA+LAMBDA) is peak estimate

[~,f1Ind] = min(abs(f-f1));
[~,f2Ind] = min(abs(f-f2));
f2Ind = f2Ind+1; %in case f1 and f2 are very close

s = abs(s);                          %not sure if this is right, but it's weird to have complex frequencies in final answer
sfilt = s(f1Ind:f2Ind,:);            %only care about s in range f1,f2
[beta,maxInds] = max(sfilt);         %get max in each column

alpha = zeros(size(beta));
lambda = zeros(size(beta));
for ii = 1:length(beta)
   alpha(ii) = s(maxInds(ii)-1+f1Ind-1,ii);   %access freqs in s not sfilt so it's less likely your out of range
   lambda(ii) = s(maxInds(ii)+1+f1Ind-1,ii);
end
alpha = 20*log10(alpha);
lambda = 20*log10(lambda);
beta = 20*log10(beta);
peaks = (.5*(alpha-lambda))./(alpha-2*beta+lambda);
peaks = 10.^(peaks./20) + offsetf-1;

end

