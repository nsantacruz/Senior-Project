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



%
%WEIGHTED KNN IS ALSO LOOKING PRETTY GOOD? idk
% Discriminant Analysis
% essentially thinks the data comes from guassian distribution? but we
% definitely need to check on that assumption with p tests and stuff.
% There's code later onn for that


%
% %tic
% linClass1 = fitcdiscr(xtrain,ytrain); % trains a linear discriminant classifier
% %toc
% disp('Time it took to train linear discriminant (unnormalized)')
% meanClass1 = predict(linClass1,xtest); % the predictions of linClass1 on test data
% percrite1 = sum(meanClass1==ytest)/length(ytest) % gave me 89.9%
% %got us 75% on the first try! (of linclass i think)
% R1 = confusionmat(meanClass1,ytest);
% R1_loss = resubLoss(linClass1)
%

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

% Ensemble Classifiers

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