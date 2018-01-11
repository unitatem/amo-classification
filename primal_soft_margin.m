function [w,b,eps] = primal_soft_margin(X,Y,options)
% x = (w,b,eps)

lambda = 1;
[X_rows,X_cols] = size(X);

H = diag([ones(1,X_cols), 0, zeros(1,X_rows)]);
f = [zeros(1,X_cols), 0, (lambda/X_rows)*ones(1,X_rows)]';
tmp = diag(Y)*X;
Aineq = -[tmp -Y eye(X_rows)];
Bineq = -ones(X_rows,1);
lb = [-inf*ones(X_cols,1); -inf; zeros(X_rows,1)];

[x,~,~,output] = quadprog(H,f,Aineq,Bineq,[],[],lb,[],[],options);
output.message

w = x(1:X_cols,:);
b = x(X_cols+1,:);
eps = x(X_cols+2:X_cols+X_rows+1,:);
