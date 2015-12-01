% STFT anaysis of 50 Hz siganl present in audio signal
clc,clear all, close all;
format compact


Type_Rec0 = 'PA';
Type_Rec1 = {'Power_recordings','Audio_recordings'};
Type_Rec2 = {'Pow', 'Aud'};
numRecordingsPerGrid = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11];
gridFreqs = [60 50 60 50 50 50 50 50 60];
trainingGridLetters = 'ABCDEFGHI' ;
count = 0;
% This line makes a file that includes all of the recordings together

numGrids = length(numRecordingsPerGrid)/2; 
% We discussed splitting up the recordings into training and test

qe = 2:2:18; 
total_pow_rec = sum(numRecordingsPerGrid(qe));
n_p_test = 0; % how many power recordings per grid do we want to save as test data?
numPowTest = (total_pow_rec) - (numGrids)*(n_p_test);
numPowTrain = (total_pow_rec) - (numPowTest);

Training_recordings = cell(numPowTrain,1);
Testing_recordings = cell(numPowTest,1);

% probably this method wont work so well for audio because we only have 2
% recordings per grid
qe = 1:2:18;
total_aud_rec = sum(numRecordingsPerGrid(qe));  %number of total audio recordings
n_a_test = 1; % how many audio recordings per grid do we want to use as test data?
numAudTest = (total_aud_rec) - (numGrids)*(n_a_test);
numAudTrain = (total_aud_rec) - (numAudTest);

Aud_Training_recordings = cell(numAudTrain,1);
Aud_Testing_recordings = cell(numAudTest,1);



TrainAudfeatmat = []; %final matrix for holding audio features for training
TestAudfeatmat = []; %final matrix for holding audio features for testing
TrainPowfeatmat = []; %final matrix for holding power features for training
TestPowfeatmat = []; %final matrix for holding power features for testing
TestAudclasses = []; %classes for the Aud testing data
TestPowclasses = []; %classes for the Pow testing data

numFeat = 7; % n = number of features
counts = 0;
trainDat = [];%temp matrix used to hold features for training
testDat = []; % temp matrix used to hold features for testing
county = 0;
featmatsize = 0;
lengthx = 0;


%for kk = 1:length(numRecordingsPerGrid)
h = waitbar(0,'Initializing waitbar...');
for kk = 2:2:length(numRecordingsPerGrid) % right now i am only looking at power recordings
    ppppp = kk/length(numRecordingsPerGrid);
    waitbar(ppppp,h,'percent done')
    endofii = numRecordingsPerGrid(kk); % this tells us the number or recordings we have for that grid (A,B...) and type (Audio, Power)
        
    if endofii==2
        numTrainGrids = endofii- n_a_test; % number of recordings which will be training data, n_a_test will be testing
        numTestGrids = n_a_test;
    else 
        numTrainGrids = endofii - n_p_test;
        numTestGrids = n_p_test;
    end

    rec_vec = 1:endofii;  % makes a vector from 1 to the number of recordings for that grid (i.e. for grid a it will be 1,2,...8,9
    if false
        r = randperm(endofii);   % randomize the total number of recordings so that which recordings we remove is random
        holdout_rec = rec_vec (r(1:numTestGrids)); %remove the number of recordings we want for holding out for testing
        data_rec = rec_vec (r(numTestGrids+1:end)); % keep the rest of the recordings as data
    else 
        data_rec = rec_vec;
    end 


    % can also use sample data function to do this...? this is good if
    % we want to select at random, but is it also good if we want to
    % split the data (so randomly select some for training, and choose
    % the rest for testing)??

    % rec_vec = 1:endofii;
    % holdout_rec = datasample(rec_vec,numTestGrids);
    % data_rec = datasample(rec_vec,num
        
         

    recType = Type_Rec0(mod(kk,2)+1);
    recTypeName = Type_Rec1{mod(kk,2)+1};
    recording_file_name = Type_Rec2{mod(kk,2)+1};
    trainingGrid = trainingGridLetters(ceil(kk/2));

    tempCell = cell(endofii - numTestGrids,1); % temp cell to hold training data
    tempCell1 = cell(numTestGrids,1); % temp cell to hold testing data



    feature_mat = [];
    feature_mat_test = [];

    %features for training
    %tempTest_Tr= cell(1,numTrainGrids); % temp used to hold reshaped(windowed) version of our file x
      
    % now we make a matrix for training data
    for ii = data_rec(1:end) % does this for loop work?

        tempStr = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
        disp(tempStr);

        if true
            [x, fs] = audioread(tempStr);  % get the samples of the .wav file
            w_length = 1000;
        else
            [tdmf,x] = recoverENF(trainingGrid,recording_file_name,ii);
            w_length = 100;
        end



        %lengthx = lengthx + length(x)
        %count = count+ 1;


        xTrunc = x(1:end-mod(length(x),w_length));
        disp(['xTrunc length ' int2str(length(xTrunc))]);
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


            feature_set = feature_set';


         if mod(kk,2) == 0 % check if this line is right
           feature_set = [((zeros(length(feature_set),1))+kk)/2, feature_set];
            %Powclasses = [Powclasses; (zeros(length(PowfeatReal),1)+ceil(ii/2))];
         end


         feature_mat = [feature_mat; feature_set];
         %size(feature_mat)
          %now I add the class number as the first column to the PowfeatReal
          
          count = count + 1;
    
   end %ends the training data loop
   
  
   trainDat = [trainDat; feature_mat];
  
if false

   % now we make a matrix for testing data
   for ii = holdout_rec(1:end) % does this for loop work?

        tempStr1 = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
        disp(tempStr1);
        if true
            [x, fs] = audioread(tempStr1);  % get the samples of the .wav file
            w_length = 1000;
        else
            [tdmf,x] = recoverENF(trainingGrid,recording_file_name,ii);
            w_length = 100;

        end


        xTrunc = x(1:end-mod(length(x),w_length));


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
        % Slope Sign Change

        FFT_Te = fft(tempTest_Te);
        % FFT_Tr = abs(FFT_Tr)/max(abs(FFT_Tr));
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
                SSI_Te(1:end-1);Var_Te(1:end-1);RMS_Te(1:end-1);WL_Te(1:end-1);...
                MeanFreq_Te(1:end-1);MedFreq_Te(1:end-1);MAVFreq_Te(1:end-1);...
                MAVSFreq_Te;MaxFreq_Te(1:end-1);VarFreq_Te(1:end-1);RMSFreq_Te(1:end-1)];



        feature_set_test = feature_set_test';
    
    
    
    
    

        if mod(kk,2) == 0 % check if this line is right
           feature_set_test = [((zeros(length(feature_set_test),1))+kk)/2, feature_set_test];
            %Powclasses = [Powclasses; (zeros(length(PowfeatReal),1)+ceil(ii/2))];
        end   

        feature_mat_test = [feature_mat_test; feature_set_test];       

        end


        testDat = [testDat; feature_mat_test];
        %    feat_mat_test_predictor = testDat(:,2:end);
        %    feat_meat_test_response = testDat(:,1);
end 
end 
xtrain = trainDat(:,2:end); %training data for testing)
ytrain = trainDat(:,1);
if false
xtest = testDat(:,2:end);
ytest = testDat(:,1);
end
%xtrain = normFeatures(xtrain);
%xtest = normFeatures(xtest);

    
    


save xtrain2holdout.mat xtrain
save ytrain2holdout.mat ytrain
if false
save xtest2holdout.mat xtest
save ytest2holdout.mat ytest
end 


