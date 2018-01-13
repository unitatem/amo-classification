close all;

%% load data
% read = true;
if read
    all_data = import_data('data/transfusion.data');
    
    %  1 => positive
    % -1 => negative
    all_data{:,5} = all_data{:,5} .* 2 - 1;
end
read=false;

data_labels = table;
data_labels.POSITIVE = 1;
data_labels.NEGATIVE = -1;

%% preprocessing
reqested_data_length = 50;
[data_positive,data_negative,validation_positive,validation_negative] =...
    extract_training_validation_data(all_data,data_labels,reqested_data_length);

data_training = [data_positive;data_negative];
data_training_rows = size(data_positive,1)+size(data_negative,1);
X = data_training(:,1:4);
Y = data_training(:,5);


%% processing
% Any hyperplane can be written as the set of points x satisfying:
% w*x-b = 0
% x = quadprog(H,f,A,b) minimizes 1/2*x'*H*x + f'*x
% subject to the restrictions A*x ≤ b.

options.Diagnostics = 'on';
options.Display = 'iter-detailed';

% solve
% [w,b,~] = primal_hard_margin(X,Y,options);
% [w,b,eps] = primal_soft_margin(X,Y,options);
% [w,b,~] = dual_hard_margin(X,Y,options);
% [w,b,x] = dual_soft_margin(X,Y,options);
[w,b,~] = augmented_lagrange(X,Y,options);

% verify
[success_rate_positive,success_rate_negative,success_rate_total] =...
    validate(data_positive,data_negative,w,b);
fprintf("Verification: pos = %.4f neg = %.4f, total = %.4f\n",...
    success_rate_positive,success_rate_negative,success_rate_total);

% crossvalidation
[validation_rate_positive,validation_rate_negative,validation_rate_total] =...
    validate(validation_positive,validation_negative,w,b);
fprintf("Validation: pos = %.4f neg = %.4f, total = %.4f\n",...
    validation_rate_positive,validation_rate_negative,validation_rate_total);

%% postprocessing
% 2D
% figure
% gplotmatrix(samples,[],theClass,['r' 'b' 'g' 'c'],[],[],false);
% sample_names = {'Recency'; 'Frequency'; 'Monetary'; 'Time'};
% text(linspace(0.1,0.85,samples_size_groups), repmat(-.1,1,4), sample_names, 'FontSize',8);
% text(repmat(-.12,1,4), linspace(0.8,0.05,samples_size_groups), sample_names, 'FontSize',8, 'Rotation',90);

% 3D
figure;
hold on;
grid on;
label = [data_labels.NEGATIVE,data_labels.POSITIVE];
mark = ["bx", "rx"];
for i = 1:length(label)
    idx = (Y == label(i));
    plot3(X(idx,1),...
          X(idx,3),...
          X(idx,4),...
          mark(i));
end

% coeff(4) = 0;
% coeff(1) = b/w(1);
% % coeff(2) = b/w(2);
% % coeff(3) = b/w(3);
% coeff(4) = b/w(4);
% a=diag(coeff);
% surf(a(1,[1 2 4]),...
%       a(2,[1 2 4]),...
%       a(3,[1 2 4]));
