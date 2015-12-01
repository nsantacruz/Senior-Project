clc, clear all, close all;
format compact

trainDat = [];%temp matrix used to hold features for training
feature_mat = [];

xtrunclength = [];
xtrunctemp = [];
numTestingRecs = 50;
count = 1;
for ii = 1:numTestingRecs % does this for loop work?

    tempStr = ['IEEEDataset/Practice_dataset/Practice_' int2str(ii) '.wav'];

    disp(tempStr);

    if true
        [x, fs] = audioread(tempStr);  % get the samples of the .wav file
        w_length = 1000;
    else
        [tdmf,x] = recoverENF(trainingGrid,recording_file_name,ii);
        w_length = 100;
    end
 
    xTrunc = x(1:end-mod(length(x),w_length));
    disp(['xTrunc length ' int2str(length(xTrunc))]);
    
    if isempty(xtrunclength)
        xtrunctemp = length(xTrunc);
        xtrunclength = [xtrunctemp];
    else
        xtrunctemp = xtrunclength(count-1)+length(xTrunc);
        xtrunclength = [xtrunclength, xtrunctemp];
    end

    count = count + 1;  
    tempTest_Tr = reshape(xTrunc,[],length(xTrunc)/w_length); % before it was just w_length
    %Integrated ENF Signal
    IENF_Tr = sum(abs(tempTest_Tr));%,2);
    %Mean Absolute Value
    MAV_Tr = mean(abs(tempTest_Tr));%,2);
    % Mean Absolute Value Slope - gives the difference beween MAVs of adjacent
    % segments
    MAVS_Tr = diff(MAV_Tr);
    %Simple Square Integral - total power per window
    SSI_Tr = sum(abs(tempTest_Tr).^2);%,2);
    %Variance
    Var_Tr = var(tempTest_Tr,0);%,%,2); %0 makes this unweighted, so it is equal to var(x)
    %RMS - root mean square
    RMS_Tr = sqrt(mean(tempTest_Tr.^2));%,2));
    % Waveform Length - cumulative length of the waveform over the time segment
    WL_Tr = sum(abs(diff(tempTest_Tr)));%,2);
    %tttt = length(WL_Tr);
    % Log Range
    LogRange_Tr = log(range(tempTest_Tr));
    % Log Variance
    LogVar_Tr = log(var(tempTest_Tr));
    
    % Log Variances of Wavelet Decomposition, 1 thru 9 levels
    WaveC1_Tr = wavedec(tempTest_Tr, 1, 'db1');
    WaveC1_Tr = WaveC1_Tr(1:length(x));
    WaveC1_Tr = reshape(WaveC1_Tr,[],round(length(WaveC1_Tr)/w_length));
    LVWave1_Tr = log(var(WaveC1_Tr));
    
    WaveC3_Tr = wavedec(tempTest_Tr, 3, 'db1');
    WaveC3_Tr = WaveC3_Tr(1:length(x));
    WaveC3_Tr = reshape(WaveC3_Tr,[],round(length(WaveC3_Tr)/w_length));
    LVWave3_Tr = log(var(WaveC3_Tr));
    
    WaveC5_Tr = wavedec(tempTest_Tr, 5, 'db1');
    WaveC5_Tr = WaveC5_Tr(1:length(x));
    WaveC5_Tr = reshape(WaveC5_Tr,[],round(length(WaveC5_Tr)/w_length));
    LVWave5_Tr = log(var(WaveC5_Tr));
    
    WaveC7_Tr = wavedec(tempTest_Tr, 7, 'db1');
    WaveC7_Tr = WaveC7_Tr(1:length(x));
    WaveC7_Tr = reshape(WaveC7_Tr,[],round(length(WaveC7_Tr)/w_length));
    LVWave7_Tr = log(var(WaveC7_Tr));
    
    WaveC9_Tr = wavedec(tempTest_Tr, 9, 'db1');
    WaveC9_Tr = WaveC9_Tr(1:length(x));
    WaveC9_Tr = reshape(WaveC9_Tr,[],round(length(WaveC9_Tr)/w_length));
    LVWave9_Tr = log(var(WaveC9_Tr));
    
    FFT_Tr = fft(tempTest_Tr);
    % FFT_Tr = abs(FFT_Tr)/max(abs(FFT_Tr));
    % Mean Frequency
    MeanFreq_Tr = meanfreq(tempTest_Tr);
    % Median Frequency
    MedFreq_Tr = medfreq(tempTest_Tr);
    % Mean Absolute Value - Frequency
    MAVFreq_Tr = mean(abs(FFT_Tr));
    % Mean Absolute Value Slope - Frequency
    MAVSFreq_Tr = diff(MAVFreq_Tr);
    % Maximum Frequency
    MaxFreq_Tr = max(abs(FFT_Tr));
    % Frequency Variance
    VarFreq_Tr = var(FFT_Tr);
    % Frequency RMS
    RMSFreq_Tr = abs(sqrt(mean(FFT_Tr.^2)));
    
    feature_set = [IENF_Tr(1:end-1);MAV_Tr(1:end-1);MAVS_Tr;...
        SSI_Tr(1:end-1);Var_Tr(1:end-1);RMS_Tr(1:end-1);WL_Tr(1:end-1);LogRange_Tr(1:end-1);...
        LogVar_Tr(1:end-1);LVWave1_Tr(1:end-1);LVWave3_Tr(1:end-1);LVWave5_Tr(1:end-1);...
        LVWave7_Tr(1:end-1);LVWave9_Tr(1:end-1);MeanFreq_Tr(1:end-1);MedFreq_Tr(1:end-1);MAVFreq_Tr(1:end-1);...
        MAVSFreq_Tr;MaxFreq_Tr(1:end-1);VarFreq_Tr(1:end-1);RMSFreq_Tr(1:end-1)];
    
     feature_set = feature_set';
     feature_mat = [feature_mat; feature_set];

end %ends the training data loop
xtrunclength = xtrunclength./w_length;
disp(xtrunclength)   
  
trainDat = [trainDat; feature_mat];
xtesting = trainDat; %training data for testing
xtestingNorm = normFeatures(xtesting);

save xtrunclength.mat xtrunclength
save xtesting.mat xtesting
save xtestingNorm.mat xtestingNorm


