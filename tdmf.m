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

[di,dc] = kmeans(abs(detrended)',2,'Replicates',20);
dc = sort(dc);


%detrendedThreshI = detrended > thresh | detrended < -thresh;
detrendedThreshI = di == 2 | di == 3;              %all of the bad indices are when the class is 2


if true
figure;
hold on;
plot(1:length(detrended),zeros(size(detrended))+dc(2),'r');
plot(1:length(detrended),zeros(size(detrended))-dc(2),'r');
plot(detrended,'b');
hold off;
end

out = in;
out(detrendedThreshI) = med(detrendedThreshI);

end

