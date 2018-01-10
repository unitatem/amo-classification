close all;

%% load data
% read = true;
if read
    all_data = import_data('transfusion.data');
    
    %  1 => positive
    % -1 => negative
    all_data{:,5} = all_data{:,5} .* 2 - 1;
end
read=false;

POSITIVE = 1;
NEGATIVE = -1;

%% preprocessing
all_data_positive = all_data{all_data{:,5} == POSITIVE,:};
all_data_negative = all_data{all_data{:,5} == NEGATIVE,:};

all_data_positive_rows = size(all_data_positive,1);
all_data_negative_rows = size(all_data_negative,1);

count = 5;
step = ceil(all_data_positive_rows / count);
data_positive = all_data_positive(step:step:all_data_positive_rows,:);
step = ceil(all_data_negative_rows / count);
data_negative = all_data_negative(step:step:all_data_negative_rows,:);

data_training = [data_positive;data_negative];
data_training_rows = size(data_positive,1)+size(data_negative,1);
X = data_training(:,1:4);
Y = data_training(:,5);

step = ceil(all_data_positive_rows / count);
assert(step > 1);
validation_positive = all_data_positive(step-1:step:all_data_positive_rows,:);
step = ceil(all_data_negative_rows / count);
assert(step > 1);
validation_negative = all_data_negative(step-1:step:all_data_negative_rows,:);

%% processing
% Any hyperplane can be written as the set of points vect_x satisfying:
% w*x-b = 0

% solving
m = 4;
n = data_training_rows;
lambda = 1;

% z = (x,b,eps)

% soft-margin
H = diag([ones(1,m), 0, zeros(1,n)]);
f = [zeros(1,m), 0, (lambda/n)*ones(1,n)]';
p = diag(Y)*X;
A = -[p -Y eye(n)];
B = -ones(n,1);
lb = [-inf*ones(m,1); -inf; zeros(n,1)];

% hard-margin
% H = diag([ones(1,m), 0]);
% f = [zeros(1,m), 0]';
% p = diag(Y)*X;
% A = -[p -Y];
% B = -ones(n,1);
% lb = [-inf*ones(m,1); -inf];

% x = quadprog(H,f,A,b) minimizes 1/2*x'*H*x + f'*x
% subject to the restrictions A*x ≤ b.
options = optimoptions('quadprog',...
    'Diagnostics','on',...
    'Display','iter-detailed');
[z,~,~,output] = quadprog(H,f,A,B,[],[],lb,[],[],options);
output.message

w = z(1:m,:);
b = z(m+1,:);
eps = z(m+2:m+n+1,:);

% verify
tmp = Y.*(X*w - ones(data_training_rows,1)*b);
s=sign(tmp);
data_positive_rows = size(data_positive,1);
data_negative_rows = size(data_negative,1);
is_positive_correct = sum(s(1:data_positive_rows)==1);
is_negative_correct = sum(s(data_positive_rows+1:data_training_rows)==1);

success_rate_positive = is_positive_correct/data_positive_rows;
success_rate_negative = is_negative_correct/data_negative_rows;
success_rate_total = (is_positive_correct+is_negative_correct)/data_training_rows;
fprintf("Verification: pos = %.4f neg = %.4f, total = %.4f\n",...
    success_rate_positive,success_rate_negative,success_rate_total);

% crossvalidation
validation_positive_rows = size(validation_positive,1);
tmp = validation_positive(:,5).*(validation_positive(:,1:4)*w - ones(validation_positive_rows,1)*b);
s=sign(tmp);
is_positive_correct = sum(s==1);

validation_negative_rows = size(validation_negative,1);
tmp = validation_negative(:,5).*(validation_negative(:,1:4)*w - ones(validation_negative_rows,1)*b);
s=sign(tmp);
is_negative_correct = sum(s==1);

validation_rate_positive = is_positive_correct/validation_positive_rows;
validation_rate_negative = is_negative_correct/validation_negative_rows;
validation_rate_total =...
    (is_positive_correct+is_negative_correct)/(validation_positive_rows+validation_negative_rows);
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
label = [NEGATIVE,POSITIVE];
mark = ["bx", "rx"];
for i = 1:length(label)
    idx = (Y == label(i));
    plot3(X(idx,1),...
          X(idx,3),...
          X(idx,4),...
          mark(i));
end
% 
% coeff(4) = 0;
% coeff(1) = b/w(1);
% % coeff(2) = b/w(2);
% % coeff(3) = b/w(3);
% coeff(4) = b/w(4);
% a=diag(coeff);
% surf(a(1,[1 2 4]),...
%       a(2,[1 2 4]),...
%       a(3,[1 2 4]));
