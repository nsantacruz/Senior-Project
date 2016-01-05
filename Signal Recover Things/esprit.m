function [ enf ] = esprit( fs,decf,framet,overt,sig,M )
%ESPRIT - estimates enf for time domain signal sig
%fs - sampling freq
%decf - decimation coef, 1 for no decimation
%framet - time, in seconds, for each window used in algo
%overt - time, in seconds, for overlap b/w windows

% short summary of algorithm?


wlen = floor(framet*(fs/decf));
olen = floor(overt*(fs/decf));

decsig = decimate(sig,decf);   % decimate from 1kHz to 250Hz
decsig = decsig(1:floor(length(decsig)/olen)*olen);  %truncate so that its divisible by wlen


enf = zeros(1,length(decsig)/olen-1);

%xframed = zeros(decsig/wlen,wlen);
for ii = 1:length(decsig)/olen-1 
    start = (ii-1)*olen+1;
    x = decsig(start:start + wlen - 1);
    %xframed(ii,:) = x;
    
    N = length(x);    % length of signal
    %M = round(N/3 + 20);
    P = 2;

    %create X matrix
    X = zeros(M,N);
    for jj = 1:N-M;
        X(:,jj) = x(jj:jj+M-1);
    end
    
    [U,~,~] = svd(X);
    
%     singVals = diag(S);
%     [~,singValsInds] = sort(singVals,'descend');
%     
%     Us = zeros(M,P);
%     for jj = 1:P
%         Us(:,jj) = U(:,singValsInds(jj));
%     end
    Us = U(:,1:P);
    U1 = Us(1:end-1,:);
    U2 = Us(2:end,:);
    
    Q = pinv(U1)*U2;
    
    effs = (fs/decf)*angle(eig(Q))/(2*pi);
    enf(ii) = effs(1);
    %f2s(ii) = effs(2);
    %f3s(ii) = sum(effs);
    
end


%figure;plot(f2s);
%figure;plot(f3s);

end

