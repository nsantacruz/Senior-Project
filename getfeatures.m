function [ feature_mat, ind ] = getfeatures( signals )
    tic
    Type_Rec0 = 'PA';
    Type_Rec1 = {'Power_recordings','Audio_recordings'};
    Type_Rec2 = {'Pow', 'Aud'};
    numRecordingsPerGrid = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11];
    %vgridFreqs = [60 50 60 50 50 50 50 50 60];
    trainingGridLetters = 'ABCDEFGHI' ;
    % This line makes a file that includes all of the recordings together
    numGrids = length(numRecordingsPerGrid)/2; 


    trainDat = []; % temp matrix used to hold features for training
    gridCounter = 1;
    ind = [];
    %for kk = 1:length(numRecordingsPerGrid)
    for kk = 2:2:length(numRecordingsPerGrid) 
        % right now i am only looking at power recordings
        endofii = numRecordingsPerGrid(kk); 
        % this tells us the number or recordings we have for that grid (A,B...) 
        % and type (Audio, Power)

        recType = Type_Rec0(mod(kk,2)+1);
        recTypeName = Type_Rec1{mod(kk,2)+1};
        recording_file_name = Type_Rec2{mod(kk,2)+1};
        trainingGrid = trainingGridLetters(ceil(kk/2));

        tempCell = cell(endofii,1); % temp cell to hold training data
        
        w_length = 10000; % length of the window

        feature_mat = [];
        % now we make a matrix for training data
        recordingCounter = 1;
        for ii = 1:endofii 
           if nargin('getfeatures') == 0
            tempStr = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
            [x, fs] = audioread(tempStr);  % get the samples of the .wav file
            w_length = 1000;
           elseif nargin('getfeatures') ~= 0
            tempStr = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
            [x, fs] = audioread(tempStr);  % get the samples of the .wav file
            w_length = 1000;% might want to change w_length for noahs files?
           end

            tempTest_Tr = reshape(x,[],length(x)/w_length); % before it was just w_length
            %Integrated ENF Signal
            IENF_Tr = sum(abs(tempTest_Tr));
            %Mean Absolute Value
            MAV_Tr = mean(abs(tempTest_Tr));
            % Mean Absolute Value Slope - gives the difference beween MAVs of adjacent
            % segments
            MAVS_Tr = diff(MAV_Tr);
            %Simple Square Integral - total power per window
            SSI_Tr = sum(abs(tempTest_Tr).^2);
            %Variance
            Var_Tr = var(tempTest_Tr,0);%,%,2); %0 makes this unweighted, so it is equal to var(x)
            %RMS - root mean square
            RMS_Tr = sqrt(mean(tempTest_Tr.^2));
            % Waveform Length - cumulative length of the waveform over the time segment
            WL_Tr = sum(abs(diff(tempTest_Tr)));

            % FFT
            FFT_Tr = fft(tempTest_Tr);
            % FFT_Tr = abs(FFT_Tr)/max(abs(FFT_Tr));
            % Mean Frequency**************
            MeanFreq_Tr = meanfreq(tempTest_Tr);
            % Median Frequency************
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
            tempind = size(feature_set,1);
            ind = [ind, ind + tempind];

            if mod(kk,2) == 0 % check if this line is right
                feature_set = [((zeros(length(feature_set),1))+kk)/2, feature_set];
            end        
            feature_mat = [feature_mat; feature_set];
            recordingCounter = recordingCounter + 1;
        end  
        gridCounter = gridCounter + 1;
    
    end 


%     Powtrain = array2table(feature_mat,'VariableNames', {'Classes', 'IENF','MAV',...
%         'MAVS','SSI','Var','RMS','WL', 'MeanFreq', 'MedFreq', 'MAVFreq',...
%         'MAVSFreq', 'MaxFreq', 'VarFreq', 'RMSFreq'});
    % save Classifier_data.mat Powtrain
    toc    
end

