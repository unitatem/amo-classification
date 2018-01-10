function [x] = primal_hard_margin(X,Y,options)
% x = (x,b)

[X_rows,X_cols] = size(X);

H = diag([ones(1,X_cols), 0]);
f = [zeros(1,X_cols), 0]';
tmp = diag(Y)*X;
A = -[tmp -Y];
B = -ones(X_rows,1);
lb = [-inf*ones(X_cols,1); -inf];

[x,~,~,output] = quadprog(H,f,A,B,[],[],lb,[],[],options);
output.message
