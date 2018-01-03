%% preprocessing
all_data = import_data('transfusion.data');
samples = all_data{:,1:4};
theClass = all_data{:,5};
% 1 => positive
% -1 => negative
theClass = theClass .* 2 - 1;

[~,no_sample_grpups] = size(samples);

% plot(samples(:,1), theClass, 'rx');
% grid on;

%% processing
%Train the SVM Classifier
cl = fitcsvm(samples,theClass, ...
    'KernelFunction','my_sigmoid',...
    'Standardize',true, ...
    'ClassNames',[-1,1]);

% Predict scores over the grid
no_cells = 10;
direction(no_cells,no_sample_grpups) = 0;
for s = 1:no_sample_grpups
    direction(:,s) = linspace(min(samples(:,s)),...
                              max(samples(:,s)),...
                              no_cells);
end
% TODO
[x1Grid,x2Grid,x3Grid] = meshgrid(direction(:,1),direction(:,2),direction(:,3));
xGrid = [x1Grid(:),x2Grid(:),x3Grid(:)];
[~,scores] = predict(cl,xGrid);

%% postprocessing
% plot the data and the decision boundary
figure
gplotmatrix(samples,[],theClass,['r' 'b' 'g' 'c'],[],[],false);
sample_names = {'Recency'; 'Frequency'; 'Monetary'; 'Time'};
text(linspace(0.1,0.85,no_sample_grpups), repmat(-.1,1,4), sample_names, 'FontSize',8);
text(repmat(-.12,1,4), linspace(0.8,0.05,no_sample_grpups), sample_names, 'FontSize',8, 'Rotation',90);


figure;
plot3(samples(:,1),samples(:,2),samples(:,3)); %,...
                  %theClass);
hold on
plot(samples(cl.IsSupportVector,1),...
     samples(cl.IsSupportVector,2),...
     samples(cl.IsSupportVector,3),...
     'ko',...
     'MarkerSize',10);
        
title('Scatter Diagram with the Decision Boundary')
contour3(x1Grid,x2Grid,x3Grid,...
        reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend({'-1','1','Support Vectors'},'Location','Best');
hold off

% print missclassification rate
CVMdl2 = crossval(cl);
misclass2 = kfoldLoss(CVMdl2);
misclass2
