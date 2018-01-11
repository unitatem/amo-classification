function [w,b,eps] = primal_hard_margin(X,Y,options)
eps=[];
% x = (w,b)

[X_rows,X_cols] = size(X);

H = diag([ones(1,X_cols), 0]);
f = [zeros(1,X_cols), 0]';
tmp = diag(Y)*X;
Aineq = -[tmp -Y];
Bineq = -ones(X_rows,1);
lb = [-inf*ones(X_cols,1); -inf];

[x,~,~,output] = quadprog(H,f,Aineq,Bineq,[],[],lb,[],[],options);
output.message

w = x(1:X_cols,:);
b = x(X_cols+1,:);
