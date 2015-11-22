function [filteredSignal] = filterENF(signal,f,df,decFactor)
% takes in signal, decimates by decFator and returns bandpass
% f is mid freq, df is deviation frequency s.t. filteredSignal is in f +/-
% df
    %A = 60, C = 60
    bOrd = 3;       
    fs = 1000;
    wlen = 8192;        
    n = round(log2(wlen));
    nfft = 2^n;
    fn = fs/2;
    f1 = (f-df)*decFactor; f2 = (f+df)*decFactor;
    w1 = f1/fn; w2 = f2/fn;
    Wp = [w1 w2];

    %ad = decimate(a,3);
    signal = decimate(signal,decFactor);

    [b,a] = butter(bOrd,Wp);
    filteredSignal = filter(b,a,signal);
    %PLOTTING
    if true
        figure;
        spectrogram(signal,wlen,4096,nfft,fs,'yaxis');
        title('Spectrogram of Power Recording');
        colormap(jet);
        figure;
        spectrogram(filteredSignal,wlen,4096,nfft,fs,'yaxis');
        title('Spectrogram of Filtered Signal');
        figure;
        startind = 2e5;
        plot(filteredSignal(startind:startind+1e4));
        title('Sample of time domain filtered signal');
        figure;
        plot(signal(startind:startind+1e4));
        title('Sample of time domain original signal');
    end
end

