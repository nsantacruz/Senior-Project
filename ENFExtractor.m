clc, clear, close all;
datasetroot = 'matlabDatasets';

%A = 60, C = 60
a = load([datasetroot '/A_Aud.mat']);
p = load([datasetroot '/A_Pow.mat']);


a = p;
bOrd = 3;
decFactor = 3;
wlen = 8192;        
n = round(log2(wlen));
nfft = 2^n;
fs = 1000;
fn = fs/2;
f1 = 49.5*decFactor; f2 = 50.5*decFactor;
w1 = f1/fn; w2 = f2/fn;
Wp = [w1 w2];

asig1 = a.tempCell{2};
psig1 = p.tempCell{2};

%ad = decimate(a,3);
asig1 = decimate(asig1,decFactor);

[b,a] = butter(bOrd,Wp);
af = filter(b,a,asig1);
if true
spectrogram(asig1,wlen,4096,nfft,fs,'yaxis');
figure;
spectrogram(af,wlen,4096,nfft,fs,'yaxis');
figure;
startind = 2e5;
plot(af(startind:startind+1e4));
figure;
plot(asig1(startind:startind+1e4));
end


