%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Short-Time Fourier Transform            %
%               with MATLAB Implementation             %
%                                                      %
% Author: M.Sc. Eng. Hristo Zhivomirov       12/21/13  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Modified by Howie Chen for ENF extraction %

function [stft3, f, t] = stft3(x, wlen, h, zfac, fs)
% function: [stft_2, f, t] = stft_2(x, wlen, h, nfft, fs)
% x - signal in the time domain
% wlen - length of the hamming window
% h - hop size
% pad - zero padding factor, default is 1 (factor of 2)
% hfact - sample interval multiplication factor
% nfft - number of FFT points
% zfac - zero padding factor
% fs - sampling frequency, Hz
% f - frequency vector, Hz
% t - time vector, s
% stft - STFT matrix (only unique points, time across columns, freq across rows)

nfft = zfac*wlen;

% represent x as column-vector if it is not
if size(x, 2) > 1
    x = x';
end

% length of the signal
xlen = length(x);

% form a periodic hamming window, with trailing zeros
win = rectwin(wlen);

% form the stft matrix
rown = ceil((1+nfft)/2);                         % calculate the total number of rows
coln = 1+fix((xlen-wlen)/h);        % calculate the total number of columns
stft3 = zeros(rown, coln);           % form the stft matrix

% initialize the indexes
indx = 0;
col = 1;

% perform STFT
while indx + wlen <= xlen
    % windowing
    xw = x(indx+1:indx+wlen).*win;
    
    % FFT with zero padding
    X = fft(xw, nfft);
    % update the stft matrix
    stft3(:, col) = X(1:rown);
    
    % update the indexes
    indx = indx + h;
    col = col + 1;
end


% calculate the time and frequency vectors
t = (wlen/2:h:wlen/2+(coln-1)*h)/fs;
f = (0:rown-1)*fs/(nfft);

end