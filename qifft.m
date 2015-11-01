function [ peaks, peaks_freq ] = qifft( s,f,f1,f2,fs )
%given s,f,t outputs of stft2, compute qifft
%s is spectrogram of signal
%f is vector representing fft frequency bins
%t are time bins
%f1 and f2 is band of interest
%ALGO: choice max bin freq BETA in b/w f1 and f2
%ALPHA is bin before, LAMBDA is after
%p = (.5*(ALPHA-LAMBDA))/(ALPHA-2*BETA+LAMBDA) is peak estimate

% https://ccrma.stanford.edu/~jos/parshl/parshl.pdf

[~,f1Ind] = min(abs(f-f1));
[~,f2Ind] = min(abs(f-f2));
% f2Ind = f2Ind+1; %in case f1 and f2 are very close

s = 20*log10(abs(s));                          %not sure if this is right, but it's weird to have complex frequencies in final answer
sfilt = s(f1Ind:f2Ind,:);            %only care about s in range f1,f2
[beta,maxInds] = max(sfilt);         %get max in each column
maxInds = maxInds +f1Ind-1;          %readjust maxInds so that it matches indices in s, not sfilt

alpha = zeros(size(beta));
lambda = zeros(size(beta));
for ii = 1:length(beta)
   alpha(ii) = s(maxInds(ii)-1,ii);   %access freqs in s not sfilt so it's less likely your out of range
   lambda(ii) = s(maxInds(ii)+1,ii);
end
% alpha = 20*log10(alpha);
% lambda = 20*log10(lambda);
% beta = 20*log10(beta);
p = (.5*(alpha-lambda))./(alpha-2*beta+lambda);
peaks = beta - .25*(alpha - lambda).*p;
% p = 10.^(p/20);
k_star = maxInds + p;
peaks_freq = .5*k_star*fs/length(f);

if false
    testpoint = 150;
    a = alpha(testpoint);
    b = beta(testpoint);
    l = lambda(testpoint);
    m = maxInds(testpoint);
    k = k_star(testpoint);

    pf = polyfit([m-1,m,m+1],[a,b,l],2);
    pv = polyval(pf,linspace(m-1,m+1));

    figure;hold on; 
    plot([m-1 m m+1],[a,b,l],'ob');
    plot(k,peaks(10),'or');
    plot(linspace(m-1,m+1),pv,'m');
    hold off;
end
end

