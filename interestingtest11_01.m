% STFT anaysis of 50 Hz siganl present in audio signal
clc, clear all, close all;
format compact
tic
%h = waitbar(0,'Please wait...');
%steps = 1000;
%for step = 1:steps
%for ppp = 1:3
Type_Rec0 = 'PA';
Type_Rec1 = {'Power_recordings','Audio_recordings'};
Type_Rec2 = {'Pow', 'Aud'};
numRecordingsPerGrid = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11];
gridFreqs = [60 50 60 50 50 50 50 50 60];
trainingGridLetters = 'ABCDEFGHI' ;
count = 0;
% This line makes a file that includes all of the recordings together
%All_recordings = cell(sum(numRecordingsPerGrid),1);
numGrids = length(numRecordingsPerGrid)/2; 
% We discussed splitting up the recordings into training and test
qe = 2:2:18; 
total_pow_rec = sum(numRecordingsPerGrid(qe));
n_p_test = 2; % how many power recordings per grid do we want to save as test data?
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
%%

%for kk = 1:length(numRecordingsPerGrid)
for kk = 2:2:length(numRecordingsPerGrid) % right now i am only looking at power recordings
        endofii = numRecordingsPerGrid(kk); % this tells us the number or recordings we have for that grid (A,B...) and type (Audio, Power)
        
        if endofii==2
            numTrainGrids = endofii- n_a_test; % number of recordings which will be training data, n_a_test will be testing
            numTestGrids = n_a_test;
        else 
            numTrainGrids = endofii - n_p_test;
            numTestGrids = n_p_test;
        end
        
        rec_vec = 1:endofii;  % makes a vector from 1 to the number of recordings for that grid (i.e. for grid a it will be 1,2,...8,9
        r = randperm(endofii);   % randomize the total number of recordings so that which recordings we remove is random
        holdout_rec = rec_vec (r(1:numTestGrids)); %remove the number of recordings we want for holding out for testing
        data_rec = rec_vec (r(numTestGrids+1:end)); % keep the rest of the recordings as data
         
        % can also use sample data function to do this...? this is good if
        % we want to select at random, but is it also good if we want to
        % split the data (so randomly select some for training, and choose
        % the rest for testing)??
        
%         rec_vec = 1:endofii;
%         holdout_rec = datasample(rec_vec,numTestGrids);
%         data_rec = datasample(rec_vec,num
        
        recType = Type_Rec0(mod(kk,2)+1);
        recTypeName = Type_Rec1{mod(kk,2)+1};
        recording_file_name = Type_Rec2{mod(kk,2)+1};
        trainingGrid = trainingGridLetters(ceil(kk/2));
        
        tempCell = cell(endofii - numTestGrids,1); % temp cell to hold training data
        tempCell1 = cell(numTestGrids,1); % temp cell to hold testing data
        
        w_length = 10000; % length of the window
        
        feature_mat = [];
        feature_mat_test = [];
        
        %features for training
        tempTest_Tr= cell(1,numTrainGrids); % temp used to hold reshaped(windowed) version of our file x
        IENF_Tr = cell(1,numTrainGrids);
        MAV_Tr = cell(1,numTrainGrids);
        MAVS_Tr = cell(1,numTrainGrids);
        SSI_Tr = cell(1,numTrainGrids);
        Var_Tr = cell(1,numTrainGrids);
        RMS_Tr = cell(1,numTrainGrids);
        WL_Tr = cell(1,numTrainGrids);
        
        %features for testing
        tempTest_te = cell(1,numTestGrids);
        IENF_te = cell(1,numTestGrids);
        MAV_te = cell(1,numTestGrids);
        MAVS_te = cell(1,numTestGrids);
        SSI_te = cell(1,numTestGrids);
        Var_te = cell(1,numTestGrids);
        RMS_te = cell(1,numTestGrids);
        WL_te = cell(1,numTestGrids);
        
   % now we make a matrix for training data
   for ii = data_rec(1:end) % does this for loop work?
       
    tempStr = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
    %disp(tempStr);
    
    [x, fs] = audioread(tempStr);  % get the samples of the .wav file
    AA1_max = max(abs(x));       % find maximum absolute value
    scldx = x/AA1_max;        % scale signal

    midf = gridFreqs(trainingGridLetters == trainingGrid);
    df = 3;
    decF = 3;
    AA1_test_Orig = scldx;
    AA1_test = filterENF(AA1_test_Orig,midf,1,decF);
    %AA1_test = AA1_test_Orig;

    AA1_test_len = length(AA1_test);    % length of signal
    wlen = 1000;                        % window length
    hop = wlen/4;                       % hop size
    pad = 2;                            % zero padding factor of 4
    fft_len = 1000;                     % FFT length (hop*fs*pad*factor) 

    K = sum(hamming(wlen, 'periodic'))/wlen;

    [s, f, noahsT] = stft2(AA1_test, wlen, hop, fft_len, fs);

qifftSig = qifft(s,f,noahsT,(midf-df)*decF,(midf+df)*decF,midf);
tdmfSig = tdmf(qifftSig,50,0.03);
    

w_length = 100;
modm = mod(length(qifftSig),w_length);
numpad = w_length-modm;
qifftSig = padarray(qifftSig,[0 numpad],'post');
qifftSig = qifftSig';

w_length = 1000;
    %lengthx = lengthx + length(x)
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
     %size(feature_mat)
      %now I add the class number as the first column to the PowfeatReal
    
   end %ends the training data loop
   
  
   trainDat = [trainDat; feature_mat];
  


   % now we make a matrix for testing data
   for ii = holdout_rec(1:end) % does this for loop work?
       
    tempStr1 = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
    %disp(tempStr1);
    
    [x, fs] = audioread(tempStr1);  % get the samples of the .wav file
    
    
    %[x, fs] = audioread(tempStr);  % get the samples of the .wav file
    AA1_max = max(abs(x));       % find maximum absolute value
    scldx = x/AA1_max;        % scale signal

    midf = gridFreqs(trainingGridLetters == trainingGrid);
    df = 3;
    decF = 3;
    AA1_test_Orig = scldx;
    AA1_test = filterENF(AA1_test_Orig,midf,1,decF);
    %AA1_test = AA1_test_Orig;

    AA1_test_len = length(AA1_test);    % length of signal
    wlen = 1000;                        % window length
    hop = wlen/4;                       % hop size
    pad = 2;                            % zero padding factor of 4
    fft_len = 1000;                     % FFT length (hop*fs*pad*factor) 

    K = sum(hamming(wlen, 'periodic'))/wlen;

    [s, f, noahsT] = stft2(AA1_test, wlen, hop, fft_len, fs);

qifftSig = qifft(s,f,noahsT,(midf-df)*decF,(midf+df)*decF,midf);
tdmfSig = tdmf(qifftSig,50,0.03);
    
    
    
w_length = 100;
modm = mod(length(qifftSig),w_length);
numpad = w_length-modm;
qifftSig = padarray(qifftSig,[0 numpad],'post');
qifftSig = qifftSig';
    


    w_length = 1000;
    %All_recordings{count} = x;
        %for t = 1:length(tempCell1)
            %tempCell1 = x;
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
   
 
   testDat = [testDat; feature_mat_test];
%    feat_mat_test_predictor = testDat(:,2:end);
%    feat_meat_test_response = testDat(:,1);
   
end 

xtrain = trainDat(:,2:end); %training data for testing)
ytrain = trainDat(:,1);
xtest = testDat(:,2:end);
ytest = testDat(:,1);


Powtrain = array2table(trainDat,'VariableNames', {'Classes', 'IENF','MAV','MAVS','SSI','Var','RMS','WL'});
Powtest = array2table(xtest,'VariableNames', {'IENF','MAV','MAVS','SSI','Var','RMS','WL'});
save Classifier_data.mat Powtrain
save Classifier_data.mat Powtest

% pow train is the final table used in the classifier, trainDat is the
% matrix of that 



% Audfeattest = [Audclasses  AudfeatReallyreal];
% AudFEAST = array2table(AudfeatReallyreal,'VariableNames', {'f1','f2','f3','f4','f5','f6','f7'});
% save Full_Feature_Matrix.mat AudfeatReallyreal
% save Full_Feature_Matrix.mat AudFEAST
% save Full_Feature_Matrix.mat Audfeattest


%% Now we are getting into some of the classification Ideas I worked on
%% Bagged Trees
%tic
bagTree1 = TreeBagger(20,xtrain,ytrain,'OOBPredictorImportance','on');
%toc 
%disp('Time it took to train Bagged Tree');
testBT1 = predict(bagTree1,xtest);
ClassTest = cell2mat(testBT1);              %testclass1 = cell2mat(Pruned_TB); maybe implement pruned tree? yes, but can't do that for bagged trees ensemble
numrite = sum(ClassTest==int2str(ytest));
disp('tree bagger 1');
perc_cor = (numrite/length(ytest))*100 %first time did pretty well - 91.74%



%%
%WEIGHTED KNN IS ALSO LOOKING PRETTY GOOD? idk
%% Discriminant Analysis
% essentially thinks the data comes from guassian distribution? but we
% definitely need to check on that assumption with p tests and stuff.
% There's code later onn for that


%%
% %tic
% linClass1 = fitcdiscr(xtrain,ytrain); % trains a linear discriminant classifier
% %toc
% disp('Time it took to train linear discriminant (unnormalized)')
% meanClass1 = predict(linClass1,xtest); % the predictions of linClass1 on test data
% percrite1 = sum(meanClass1==ytest)/length(ytest) % gave me 89.9%
% %got us 75% on the first try! (of linclass i think)
% R1 = confusionmat(meanClass1,ytest);
% R1_loss = resubLoss(linClass1)
%%

% Now trying Quadratic Discriminant instead of linear

%I used 'pseudoQuadradic' because there were issues with singularities (I
%think one or more of our features has values too close to zero) and that
%is whatthe literature I read suggests
%tic
quadClass1 = fitcdiscr(xtrain,ytrain,'DiscrimType','pseudoQuadratic');
%toc
%disp('Time it took to train (pseudo)quadratic discriminant (unnormalized)')
meanClassQ1 = predict(quadClass1,xtest);
disp('quad disc class');
percriteQ1 = sum(meanClassQ1==ytest)/length(ytest)  % got %94.6!
% first try on quadratic discriminant I got us 93%!!!!!!! lets try this again!
% this might be an error though b/c it might be not enough decimal places for the predictors
RQ1 = confusionmat(meanClassQ1,ytest);
RQ1_loss = resubLoss(quadClass1);

%resub error is the resubstitution error on the training class - not a good
%metric for seeing how performance is on testing data, but still usefull

%% Ensemble Classifiers

%tic
%ens = fitensemble(X,Y,model,numberens,learners)
ens1 = fitensemble(xtrain,ytrain,'Bag',100,'Tree','Type','Classification');
disp('ensemble bagged trees');
pred1 = sum(predict(ens1,xtest)==ytest)/length(ytest)
predErr =  resubLoss(ens1) - (1-pred1);% shows the difference between what it thinks it got (based on training),% and what it did get (based on testing)

% waitbar(step / steps)
% end
% end
% close(h) 
% toc


%%
% cens1 = crossval(ens1,'KFold',5);
% 
% figure;
% plot(loss(ens4,xtest,ytest,'mode','cumulative'));
% hold on;
% plot(kfoldLoss(cens1,'mode','cumulative'),'r.');
% hold off;
% xlabel('Number of trees');
% ylabel('Classification error');
% legend('Test','Cross-validation','Location','NE');
% title('Cross validated loss vs loss on new data predictions')