function [ f0,f1,f2,maxy ] = binAlgoMaxDFTFinder( f0,f1,f2,tsig,fs,eps )
%f0 - center of current band of interest
%f1 - left side of band of interest
%f2 - right side
%tsig - time domain signal
%fs - duh
%N - number of points in FFT
%e - convergence criterion
ns = 0:length(tsig)-1;

dft = @(f) sum(tsig.*exp((-1i*2*pi*f.*ns)./fs));

y0 = dft(f0);
y1 = dft(f1);
y2 = dft(f2);

if (f2-f1) < eps
    maxf = f0;
    maxy = y0;
    return
end

if y1 < y2
    binAlgoMaxDFTFinder(mean(f0,f2),f0,f2,tsig,fs,eps)
else
    binAlgoMaxDFTFinder(mean(f0,f1),f1,f0,tsig,fs,eps)
end





end

