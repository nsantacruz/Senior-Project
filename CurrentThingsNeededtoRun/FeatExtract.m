function [xtrain, ytrain, xtest, ytest, ybintrain, ybintest,xtltrain,xtltest] =...
    FeatExtract(gridlettrs,numrgrid, numTestRecs,isP, Norm, Est, name1, name2, name3, name4,...
    name5, name6,name7,name8)
% numTestRecs: specifies the number of recordings used for testing - the
% remaining recordings will be used for training
%
% isP: Audio or power; if 0, working with audio files; if 1, working with
% power files
%
% Norm: if 0, do not normalize features, if 1, normalize features to 1
%
% Est: if 0, do not use estimated ENF signal; if 1, use estimated ENF
%
% name1 and name2: strings containing the names of files to be saved for
% ybintrain and ybintest
%
% name3 thru name6: strings containing the names of files to be saved for
% xtrain, ytrain, xtest, and ytest

%name 7 and 8 are xtrunclengthtraining and xtrunclengthtesting

Type_Rec0 = 'PA';
Type_Rec1 = {'Power_recordings','Audio_recordings'};
numRecsPerGrid = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11];
numRecsPerGrid = numrgrid;
% numRecsPerGrid = [2,11];
% trainingGridLetters = 'ABCDEFGHI' ;
trainingGridLetters = gridlettrs;
% trainingGridLetters = '';
count = 0;
xcount = 1;
xcount2 = 1;
trainDat = []; %temp matrix used to hold features for training
testDat = []; %temp matrix used to hold features for testing
xtltrain = [];
xtltest = [];
%for kk = 1:length(numRecordingsPerGrid)
h = waitbar(0,'Initializing waitbar...');

if isP
    loopInds = 2:2:length(numRecsPerGrid);
else
    loopInds = 1:2:length(numRecsPerGrid);
end

for kk = loopInds
     waitb = kk/length(numRecsPerGrid);
     waitbar(waitb,h,'percent done')
    endofii = numRecsPerGrid(kk); % this tells us the number or recordings we have for that grid (A,B...) and type (Audio, Power)

    rec_vec = 1:endofii;  % makes a vector from 1 to the number of recordings for that grid (i.e. for grid a it will be 1,2,...8,9
   
    r = randperm(endofii);  % randomize the total number of recordings so that which recordings we remove is random
    if numTestRecs == 0
        holdout_rec = [];
        data_rec = rec_vec;
    else 
        holdout_rec = rec_vec (r(1:numTestRecs)); %remove the number of recordings we want for holding out for testing
        data_rec = rec_vec (r(numTestRecs+1:end)); % keep the rest of the recordings as data
    end
    
        
    recType = Type_Rec0(mod(kk,2)+1);
    recTypeName = Type_Rec1{mod(kk,2)+1};

    trainingGrid = trainingGridLetters(ceil(kk/2));

    feature_mat = [];
    feature_mat_test = [];
      
    % now we make a matrix for training data
    for ii = data_rec(1:end) % does this for loop work?
        tempStr = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName...
            '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
        disp('Training...')
        disp(tempStr);
        if Est
             [tdmf,x] = recoverENF(trainingGrid,recording_file_name,ii);
             w_length = 100;
        else
            [x, fs] = audioread(tempStr); % get the samples of .wav file
            w_length = 1000;
        end

        xTrunc = x(1:end-mod(length(x),w_length));
        %disp(['xTrunc length ' int2str(length(xTrunc))]);
        
    if isempty(xtltrain) %xTruncLengthTraining
        xtrunctemp = length(xTrunc);
        xtltrain= [xtrunctemp];
    else
        xtrunctemp = xtltrain(xcount-1)+length(xTrunc);
        xtltrain = [xtltrain, xtrunctemp];
    end
    
    xcount = xcount + 1;
        
        
        
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

         if mod(kk,2) == 0
           feature_set = [((zeros(length(feature_set),1))+kk)/2, feature_set];
         else
           feature_set = [((zeros(length(feature_set),1))+kk+1)/2, feature_set];
         end

         feature_mat = [feature_mat; feature_set];
         %size(feature_mat)
          %now I add the class number as the first column to the PowfeatReal          
          count = count + 1;
    
   end %ends the training data loop
   trainDat = [trainDat; feature_mat];
   % now we make a matrix for testing data
   
   if ~isempty(holdout_rec)
       for ii = holdout_rec(1:end) % does this for loop work?

            tempStr1 = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
            disp('Testing...');
            disp(tempStr1);
            if Est
                [tdmf,x] = recoverENF(trainingGrid,recording_file_name,ii);
                w_length = 100;
            else
                [x, fs] = audioread(tempStr1);  % get the samples of the .wav file
                w_length = 1000;
            end
            xTrunc = x(1:end-mod(length(x),w_length));
            %disp(['xTrunc length ' int2str(length(xTrunc))]);
        
            if isempty(xtltest) %xTruncLengthTest
                xtrunctemp = length(xTrunc);
                xtltest= [xtrunctemp];
            else
                xtrunctemp = xtltest(xcount2-1)+length(xTrunc);
                xtltest = [xtltest, xtrunctemp];
            end
    
            xcount2 = xcount2 + 1;
            
            

            tempTest_Te = reshape(xTrunc,[],length(xTrunc)/w_length);
            %Integrated ENF Signal
            IENF_Te = sum(abs(tempTest_Te));%,2);
            %Mean Absolute Value
            MAV_Te = mean(abs(tempTest_Te));%,2);
            % Mean Absolute Value Slope - gives the difference beween MAVs of adjacent
            % segments
            MAVS_Te = diff(MAV_Te);
            %Simple Square Integral - total power per window
            SSI_Te = sum(abs(tempTest_Te).^2);%,2);
            %Variance
            Var_Te = var(tempTest_Te,0);%,2); %0 makes this unweighted, so it is equal to var(x)
            %RMS - root mean square
            RMS_Te = sqrt(mean(tempTest_Te.^2));%,2));
            % Waveform Length - cumulative length of the waveform over the time segment
            WL_Te = sum(abs(diff(tempTest_Te)));%,2);
            % Log Range
            LogRange_Te = log(range(tempTest_Te));
            % Log Variance
            LogVar_Te = log(var(tempTest_Te));

            % Log Variances of Wavelet Decomposition, 1 thru 9 levels
            WaveC1_Te = wavedec(tempTest_Te, 1, 'db1');
            WaveC1_Te = WaveC1_Te(1:length(x));
            WaveC1_Te = reshape(WaveC1_Te,[],round(length(WaveC1_Te)/w_length));
            LVWave1_Te = log(var(WaveC1_Te));

            WaveC3_Te = wavedec(tempTest_Te, 3, 'db1');
            WaveC3_Te = WaveC3_Te(1:length(x));
            WaveC3_Te = reshape(WaveC3_Te,[],round(length(WaveC3_Te)/w_length));
            LVWave3_Te = log(var(WaveC3_Te));

            WaveC5_Te = wavedec(tempTest_Te, 5, 'db1');
            WaveC5_Te = WaveC5_Te(1:length(x));
            WaveC5_Te = reshape(WaveC5_Te,[],round(length(WaveC5_Te)/w_length));
            LVWave5_Te = log(var(WaveC5_Te));

            WaveC7_Te = wavedec(tempTest_Te, 7, 'db1');
            WaveC7_Te = WaveC7_Te(1:length(x));
            WaveC7_Te = reshape(WaveC7_Te,[],round(length(WaveC7_Te)/w_length));
            LVWave7_Te = log(var(WaveC7_Te));

            WaveC9_Te = wavedec(tempTest_Te, 9, 'db1');
            WaveC9_Te = WaveC9_Te(1:length(x));
            WaveC9_Te = reshape(WaveC9_Te,[],round(length(WaveC9_Te)/w_length));
            LVWave9_Te = log(var(WaveC9_Te));

            FFT_Te = fft(tempTest_Te);
            % FFT_Te = abs(FFT_Te)/max(abs(FFT_Te));
            % Mean Frequency
            MeanFreq_Te = meanfreq(tempTest_Te);
            % Median Frequency
            MedFreq_Te = medfreq(tempTest_Te);
            % Mean Absolute Value - Frequency
            MAVFreq_Te = mean(abs(FFT_Te));
            % Mean Absolute Value Slope - Frequency
            MAVSFreq_Te = diff(MAVFreq_Te);
            % Maximum Frequency
            MaxFreq_Te = max(abs(FFT_Te));
            % Frequency Variance
            VarFreq_Te = var(FFT_Te);
            % Frequency RMS
            RMSFreq_Te = abs(sqrt(mean(FFT_Te.^2)));

            feature_set_test = [IENF_Te(1:end-1);MAV_Te(1:end-1);MAVS_Te;...
            SSI_Te(1:end-1);Var_Te(1:end-1);RMS_Te(1:end-1);WL_Te(1:end-1);LogRange_Te(1:end-1);...
            LogVar_Te(1:end-1);LVWave1_Te(1:end-1);LVWave3_Te(1:end-1);LVWave5_Te(1:end-1);...
            LVWave7_Te(1:end-1);LVWave9_Te(1:end-1);MeanFreq_Te(1:end-1);MedFreq_Te(1:end-1);MAVFreq_Te(1:end-1);...
            MAVSFreq_Te;MaxFreq_Te(1:end-1);VarFreq_Te(1:end-1);RMSFreq_Te(1:end-1)];

            feature_set_test = feature_set_test';

            if mod(kk,2) == 0
               feature_set_test = [((zeros(length(feature_set_test),...
                   1))+kk)/2, feature_set_test];
            else
               feature_set_test = [((zeros(length(feature_set_test),...
                   1))+kk+1)/2, feature_set_test];   
            end      
            feature_mat_test = [feature_mat_test; feature_set_test];
        end
            testDat = [testDat; feature_mat_test];
   end 
end 
xtrain = trainDat(:,2:end); % Data for testing
ytrain = trainDat(:,1);

if isempty(testDat)
    xtest = [];
    ytest = [];
else
    xtest = testDat(:,2:end);
    ytest = testDat(:,1);   
end

if Norm
    xtrain = normFeatures(xtrain);
    xtest = normFeatures(xtest);
end

% if nargin == 10
%     save(name3, 'xtrain')
%     save(name4, 'ytrain')
%     save(name5, 'xtest')
%     save(name6, 'ytest')
% elseif nargin == 8
%     save(name3, 'xtrain')
%     save(name4, 'ytrain')
% end 
ybintrain = zeros(size(ytrain)) + isP;
ybintest = zeros(size(ytest)) + isP;
save(name1, 'ybintrain');
save(name2, 'ybintest');
save(name3, 'xtrain')
save(name4, 'ytrain')
save(name5, 'xtest')
save(name6, 'ytest')


xtltrain = xtltrain ./w_length
xtltest = xtltest./w_length

% this is to change the indices to reflect that we removed one window from
% each recording due to off-by-one errors.
minus1 = 1;
for t = 1:length(xtltrain)
    xtltrain(t) = xtltrain(t) - minus1;
    minus1 = minus1+1;
end
    
minus1 = 1;
if ~isempty(xtltest)
    for t = 1:length(xtltest)
        xtltest(t) = xtltest(t) - minus1;
        minus1 = minus1 + 1;
    end
end
    

save(name7,'xtltrain')
save(name8,'xtltest')


disp(xtltrain)   
disp(xtltest)



close(h)
end

