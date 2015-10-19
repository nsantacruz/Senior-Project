% STFT anaysis of 50 Hz siganl present in audio signal
clc, clear all, close all;
format compact
recordingTypeLetters = 'PA'
recordingTypeNames = {'Power_recordings','Audio_recordings'};
recording_file = {'Pow', 'Aud'};
numRecordingsPerGrid = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11]
trainingGridLetters = 'ABCDEFGHI' ;
count = 0;
All_recordings = cell(sum(numRecordingsPerGrid),1);
    
    for kk = 1:length(numRecordingsPerGrid)
        endofii = numRecordingsPerGrid(kk);
%         if mod(kk,2) == 0
%             %Power Grid
%             recType = 'P';
%         else
%             recType = 'A';
%         end
        
        recType = recordingTypeLetters(mod(kk,2)+1);
        recTypeName = recordingTypeNames{mod(kk,2)+1};
        recording_file_name = recording_file{mod(kk,2)+1};
        trainingGrid = trainingGridLetters(ceil(kk/2));
        
   tempCell = cell(endofii,1);
   
   for ii = 1:endofii
    tempStr = ['IEEEDataset/Grid_' trainingGrid '/' recTypeName '/Train_Grid_' trainingGrid '_' recType int2str(ii) '.wav'];
    disp(tempStr);
    
    [x, fs] = audioread(tempStr);  % get the samples of the .wav file
    count = count+ 1;
    All_recordings{count} = x;
    
    tempCell{ii} = x;

   end 
   save([trainingGrid '_' recording_file_name '.mat'],'tempCell');
    end


%[x, fs] = audioread(['Train_Grid_' trainingGrid(jj) '_' recordingTypeLetters(ii) int2str(var) '.wav']);  % get the samples of the .wav file
%[xAA2, fsAA2] = audioread('Train_Grid_A_A2.wav');






