function [w,b,mock] = augmented_lagrange(X,Y,options)
mock = [];
options.Display = 'off';

% x = (w,b)

X_rows = size(X,1);

eps = 10e-5;
rho = 1;
lambda(1:X_rows,1) = 0;

w = zeros(1,4);
b = 0;
x = [w,b];

delta = 2*eps;
iter = 0;
while delta > eps
    iter = iter + 1;
    
    constraints = @(w,b)(ones(X_rows,1)-Y.*(X*w'-b*ones(X_rows,1)));
    funImpl = @(w,b)(w*w'+sum(max(0.5*lambda+rho*constraints(w,b),0).^2));
    fun = @(x)(funImpl(x(1:4),x(5)));

    [x_new,~,~,output] = fminsearch(fun,x,options);

    w = x_new(1:4);
    b = x_new(5);

    lambda_new = max(lambda + 2*rho*constraints(w,b),0);
    if norm(lambda_new-lambda) < 0.5
        rho = 2*rho;
    end
    lambda = lambda_new;

    delta = norm(x_new-x);
    x = x_new;
    
    fprintf("Iter = %d Delta =%f\n",iter,delta);
end
output.message

w = w';
