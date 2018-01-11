function [w,b,x] = dual_hard_margin(X,Y,options)
options.MaxIterations = 10000;

X_rows = size(X,1);

H = (Y.*X)*(Y.*X)';
f = -ones(X_rows,1);
Aeq = Y';
Beq = 0;
lb = zeros(X_rows,1);
[x,~,~,output] = quadprog(H,f,[],[],Aeq,Beq,lb,[],[],options);
output.message

x(x<10e-10)=0;

w = sum(x.*Y.*X)';
tmp = find(x>0);
i = tmp(1);
b = X(i,:)*w-Y(i);
