%clc, clear all, close all
A_Pow = load('A_Pow.mat','tempCell');

AP1 = A_Pow.tempCell(1);
% AP2 = A_Pow.tempCell(2);
% AP3 = A_Pow.tempCell(3);
% AP4 = A_Pow.tempCell(4);
% AP5 = A_Pow.tempCell(5);
% AP6 = A_Pow.tempCell(6);
% AP7 = A_Pow.tempCell(7);
% AP8 = A_Pow.tempCell(8);
% AP9 = A_Pow.tempCell(9);

%%
w_length = 10000;
AP1_test = reshape(A_Pow.tempCell{1},[],w_length);
%Integrated ENF Signal
IENF = sum(abs(AP1_test),2);
%Mean Absolute Value
MAV = mean(abs(AP1_test),2);
% Mean Absolute Value Slope - gives the difference beween MAVs of adjacent
% segments
MAVS = diff(MAV);
%Simple Square Integral - total power per window
SSI = sum(abs(AP1_test).^2,2);
%Variance
Var = var(AP1_test,0,2); %0 makes this unweighted, so it is equal to var(x)
%RMS - root mean square
RMS = sqrt(mean(AP1_test.^2,2));
% Waveform Length - cumulative length of the waveform over the time segment
WL = sum(abs(diff(AP1_test)),2);
% Slope Sign Change

% MMAV1 = (1/w_length)*sum(weight*abs(AP1_test),2);
