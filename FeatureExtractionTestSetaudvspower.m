% STFT anaysis of 50 Hz siganl present in audio signal
clc,clear all, close all;
format compact


trainDat = [];%temp matrix used to hold features for training
feature_mat = [];

xtrunclength = [];
xtrunctemp = [];
numTestingRecs = 50;
count = 1;
for ii = 1:numTestingRecs 

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
    if false
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
        SSI_Tr(1:end-1);Var_Tr(1:end-1);RMS_Tr(1:end-1);WL_Tr(1:end-1);...
        MeanFreq_Tr(1:end-1);MedFreq_Tr(1:end-1);MAVFreq_Tr(1:end-1);...
        MAVSFreq_Tr;MaxFreq_Tr(1:end-1);VarFreq_Tr(1:end-1);RMSFreq_Tr(1:end-1)];

    end 
    
    
    
    IENF_Tr = sum(abs(tempTest_Tr));
    MAV_Tr = mean(abs(tempTest_Tr));
    Var_Tr = var(tempTest_Tr,0);
    FFT_Tr = fft(tempTest_Tr);
    MeanFreq_Tr = meanfreq(tempTest_Tr);
    MedFreq_Tr = medfreq(tempTest_Tr);
    MAVFreq_Tr = mean(abs(FFT_Tr));
    MaxFreq_Tr = max(abs(FFT_Tr));
    VarFreq_Tr = var(FFT_Tr);




    feature_set = [IENF_Tr(1:end-1);MAV_Tr(1:end-1);...
        Var_Tr(1:end-1);...
        MeanFreq_Tr(1:end-1);MedFreq_Tr(1:end-1);MAVFreq_Tr(1:end-1);...
       MaxFreq_Tr(1:end-1);VarFreq_Tr(1:end-1)];
        
    
        feature_set = feature_set';


     feature_mat = [feature_mat; feature_set];



end %ends the training data loop
xtrunclength = xtrunclength./w_length;
disp(xtrunclength)   
  
trainDat = [trainDat; feature_mat];
  
 
xtesting = trainDat; %training data for testing
xtestingNorm = normFeatures(xtesting);

save xtrunclengthaudpow.mat xtrunclength
save xtestingaudpow.mat xtesting
save xtestingNormaudpow.mat xtestingNorm


