function [x] = primal_soft_margin(X,Y,options)
% x = (x,b,eps)

lambda = 1;
[X_rows,X_cols] = size(X);

H = diag([ones(1,X_cols), 0, zeros(1,X_rows)]);
f = [zeros(1,X_cols), 0, (lambda/X_rows)*ones(1,X_rows)]';
tmp = diag(Y)*X;
A = -[tmp -Y eye(X_rows)];
B = -ones(X_rows,1);
lb = [-inf*ones(X_cols,1); -inf; zeros(X_rows,1)];

[x,~,~,output] = quadprog(H,f,A,B,[],[],lb,[],[],options);
output.message
