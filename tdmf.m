function [ out ] = tdmf( in,n,thresh )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
med = zeros(size(in));
for ii = 1:length(in)
    if ii <= n
        med(ii) = median(in(ii:ii+n));
    elseif ii > length(in)-n
        med(ii) = median(in(ii-n:ii));
    else
        med(ii) = median(in(ii-n:ii+n)); 
    end
end

detrended = in - med;
detrendedThreshI = detrended > thresh | detrended < -thresh;

if true
figure;
hold on;
plot(1:length(detrended),zeros(size(detrended))+thresh,'r');
plot(1:length(detrended),zeros(size(detrended))-thresh,'r');
plot(detrended,'b');
hold off;
end

out = in;
out(detrendedThreshI) = med(detrendedThreshI);

end

