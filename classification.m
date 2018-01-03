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
model = fitcsvm(samples,theClass, ...
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

% cartesian product, create grid
xGrid(no_cells^no_sample_grpups, no_sample_grpups) = 0;
xGrid_size = no_cells^(no_sample_grpups-1);
for s = 1:no_sample_grpups
    for i = 1:2^(s-1)
        for c = 1:no_cells
            for g = 1:xGrid_size
                xGrid(xGrid_size*(no_cells*(i-1)+(c-1))+g,s) = direction(c,s);
            end
        end
    end
    xGrid_size = xGrid_size/no_cells;
end

[~,scores] = predict(model,xGrid);

% crossvalidation
CVMdl2 = crossval(model);
misclass2 = kfoldLoss(CVMdl2);
misclass2

%% postprocessing
% 2D
figure
gplotmatrix(samples,[],theClass,['r' 'b' 'g' 'c'],[],[],false);
sample_names = {'Recency'; 'Frequency'; 'Monetary'; 'Time'};
text(linspace(0.1,0.85,no_sample_grpups), repmat(-.1,1,4), sample_names, 'FontSize',8);
text(repmat(-.12,1,4), linspace(0.8,0.05,no_sample_grpups), sample_names, 'FontSize',8, 'Rotation',90);

% 3D
figure;
hold on;
grid on;
label = [-1, 1];
mark = ["rx", "bx"];
for i = 1:2
    idx = (theClass == label(i));
    plot3(samples(idx,1),...
          samples(idx,2),...
          samples(idx,4),...
          mark(i));
end

plot3(samples(model.IsSupportVector,1),...
      samples(model.IsSupportVector,2),...
      samples(model.IsSupportVector,4),...
      'ko',...
      'MarkerSize',10);
        
% contour3(x1Grid,x2Grid,x3Grid,...
%          reshape(scores(:,2),size(x1Grid)),[0 0],'k');
    
legend({'-1','1','Support Vectors'},'Location','Best');
hold off


