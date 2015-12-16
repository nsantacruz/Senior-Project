% STFT anaysis of 50 Hz siganl present in audio signal
clc, clear all, close all;
format compact
recordingTypeLetters = 'PA';
recordingTypeNames = {'Power_recordings','Audio_recordings'};
recording_file = {'Pow', 'Aud'};
numRecordingsPerGrid = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11];
trainingGridLetters = 'ABCDEFGHI' ;
count = 0;
% This line makes a file that includes all of the recordings together
%All_recordings = cell(sum(numRecordingsPerGrid),1);
n_grids = length(numRecordingsPerGrid)/2; 
% We discussed splitting up the recordings into training and test
i = 2:2:18;
total_pow_rec = sum(numRecordingsPerGrid(i));
n_p_test = 2; % how many power recordings per grid do we want to save as test data?
num_pow_test = (total_pow_rec) - (n_grids)*(n_p_test);
num_pow_training = (total_pow_rec) - (num_pow_test);

Training_recordings = cell(num_pow_training,1);
Testing_recordings = cell(num_pow_test,1);

% probably this method wont work so well for audio because we only have 2
% recordings per grid
i = 1:2:18;
total_aud_rec = sum(numRecordingsPerGrid(i));  %number of total audio recordings
n_a_test = 1; % how many audio recordings per grid do we want to use as test data?
num_aud_test = (total_aud_rec) - (n_grids)*(n_a_test);
num_aud_training = (total_aud_rec) - (num_aud_test);

Aud_Training_recordings = cell(num_aud_training,1);
Aud_Testing_recordings = cell(num_aud_test,1);



TrainAudfeatmat = []; %final matrix for holding audio features for training
TestAudfeatmat = []; %final matrix for holding audio features for testing
TrainPowfeatmat = []; %final matrix for holding power features for training
TestPowfeatmat = []; %final matrix for holding power features for testing
TestAudclasses = []; %classes for the Aud testing data
TestPowclasses = []; %classes for the Pow testing data

n_feat = 7; % n = number of features
counts = 0;
feat_mat = [];%temp matrix used to hold features for training
feat_mat_test = []; % temp matrix used to hold features for testing
county = 0;
featmatsize = 0;
lengthx = 0;
%%

%for kk = 1:length(numRecordingsPerGrid)
for kk = 2:2:length(numRecordingsPerGrid) % right now i am only looking at power recordings
        endofii = numRecordingsPerGrid(kk); % this tells us the number or recordings we have for that grid (A,B...) and type (Audio, Power)
        
        if endofii==2
            num_train = endofii- n_a_test; % number of recordings which will be training data, n_a_test will be testing
            n_test = n_a_test;
        else 
            num_train = endofii - n_p_test;
            n_test = n_p_test;
        end
        
        rec_vec = 1:endofii;  % makes a vector from 1 to the number of recordings for that grid (i.e. for grid a it will be 1,2,...8,9
        r = randperm(endofii);   % randomize the total number of recordings so that which recordings we remove is random
        holdout_rec = rec_vec (r(1:n_test)); %remove the number of recordings we want for holding out for testing
        data_rec = rec_vec (r(n_test+1:end)); % keep the rest of the recordings as data
         
        recType = recordingTypeLetters(mod(kk,2)+1);
        recTypeName = recordingTypeNames{mod(kk,2)+1};
        recording_file_name = recording_file{mod(kk,2)+1};
        trainingGrid = trainingGridLetters(ceil(kk/2));
        
        tempCell = cell(endofii - n_test,1); % temp cell to hold training data
        tempCell1 = cell(n_test,1); % temp cell to hold testing data
        
        w_length = 10000;
        
        feature_mat = [];
        feature_mat_test = [];
        
        %features for training
        tempTest_Tr= cell(1,num_train);
        IENF_Tr = cell(1,num_train);
        MAV_Tr = cell(1,num_train);
        MAVS_Tr = cell(1,num_train);
        SSI_Tr = cell(1,num_train);
        Var_Tr = cell(1,num_train);
        RMS_Tr = cell(1,num_train);
        WL_Tr = cell(1,num_train);
        
        %features for testing
        tempTest_te = cell(1,n_test);
        IENF_te = cell(1,n_test);
        MAV_te = cell(1,n_test);
        MAVS_te = cell(1,n_test);
        SSI_te = cell(1,n_test);
        Var_te = cell(1,n_test);
        RMS_te = cell(1,n_test);
        WL_te = cell(1,n_test);
        
   % now we make a matrix for training data
   for ii = data_rec(1:end) % does this for loop work?
       
    tempStr = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
    disp(tempStr);
    
    [x, fs] = audioread(tempStr);  % get the samples of the .wav file
    lengthx = lengthx + length(x)
    %count = count+ 1;
    %All_recordings{count} = x;
        %for t = 1:length(tempCell)
            %tempCell{t} = x;
            %tempTest_Tr{t} = reshape(tempCell{t},[],w_length);
            tempTest_Tr = reshape(x,[],length(x)/w_length); % before it was just w_length
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
            tttt = length(WL_Tr);
            
            % Slope Sign Change
        %end 
        %now each row is a feature
     feature_set = [IENF_Tr(1:end-1);MAV_Tr(1:end-1);MAVS_Tr;SSI_Tr(1:end-1);Var_Tr(1:end-1);RMS_Tr(1:end-1);WL_Tr(1:end-1)];
     %counts = length(feature_set) + counts
     feature_set = feature_set';

     if mod(kk,2) == 0 % check if this line is right
       feature_set = [((zeros(length(feature_set),1))+kk)/2, feature_set];
        %Powclasses = [Powclasses; (zeros(length(PowfeatReal),1)+ceil(ii/2))];
     end

         
     feature_mat = [feature_mat; feature_set];
     size(feature_mat)
      %now I add the class number as the first column to the PowfeatReal
    
   end %ends the training data loop
   
   %counts
   feat_mat = [feat_mat; feature_mat];
   %featmatsize = featmatsize + size(feature_mat)
   %feature_set = {IENF,MAV,MAVS,SSI,Var,RMS,WL};
   %save([trainingGrid '_' recording_file_name '.mat'],'tempCell');



   % now we make a matrix for testing data
   for ii = holdout_rec(1:end) % does this for loop work?
       
    tempStr1 = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
    disp(tempStr1);
    
    [x, fs] = audioread(tempStr1);  % get the samples of the .wav file
    
    %All_recordings{count} = x;
        %for t = 1:length(tempCell1)
            tempCell1 = x;
            tempTest_Te = reshape(x,[],length(x)/w_length);
      
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
            
    feature_set_test = [IENF_Te(1:end-1);MAV_Te(1:end-1);MAVS_Te;SSI_Te(1:end-1);Var_Te(1:end-1);RMS_Te(1:end-1);WL_Te(1:end-1)];
    %county = length(feature_set_test) + county  
    feature_set_test = feature_set_test';
    if mod(kk,2) == 0 % check if this line is right
       feature_set_test = [((zeros(length(feature_set_test),1))+kk)/2, feature_set_test];
        %Powclasses = [Powclasses; (zeros(length(PowfeatReal),1)+ceil(ii/2))];
    end   
            
   feature_mat_test = [feature_mat_test; feature_set_test];       
            
   end
   
   %county  
   feat_mat_test = [feat_mat_test; feature_mat_test];
   feat_mat_test_predictor = feat_mat_test(:,2:end);
   feat_meat_test_response = feat_mat_test(:,1);
   
   size(feat_mat_test)
   
   end 

Pow_train = array2table(feat_mat,'VariableNames', {'Classes', 'IENF','MAV','MAVS','SSI','Var','RMS','WL'});
Pow_test = array2table(feat_mat_test_predictor,'VariableNames', {'IENF','MAV','MAVS','SSI','Var','RMS','WL'});
save Classifier_data.mat Pow_train
save Classifier_data.mat Pow_test


% Audfeattest = [Audclasses  AudfeatReallyreal];
% AudFEAST = array2table(AudfeatReallyreal,'VariableNames', {'f1','f2','f3','f4','f5','f6','f7'});
% save Full_Feature_Matrix.mat AudfeatReallyreal
% save Full_Feature_Matrix.mat AudFEAST
% save Full_Feature_Matrix.mat Audfeattest

%% DONT RUN THESE LINES TILL THE END
yfit26 = C_101.predictFcn(feat_mat_test_predictor);
test26 = length(nonzeros(yfit26==feat_meat_test_response))/length(yfit26)
%%
yfit27 = C_106.predictFcn(feat_mat_test_predictor);test27 = length(nonzeros(yfit27==feat_meat_test_response))/length(yfit27)
