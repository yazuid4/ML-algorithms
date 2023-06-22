function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


n = length(theta); 

reg_terme = 0;

for i=2:n,
      reg_terme = reg_terme + (lambda/(2*m))*(theta(i)^2);
end

for i=1:m,
J=J+(1/m)*(-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
end

J=J+reg_terme;

     
  for i=1:m,
    grad(1)=grad(1)+(1/m)*(sigmoid(X(i,:)*theta) - y(i)) *X(i,1);
   end
    
for j=2:n
    for i=1:m
       grad(j)=grad(j)+(1/m)*( sigmoid(X(i,:)*theta)- y(i) )*X(i,j);
    end
      grad(j) = grad(j)+(lambda/m)*theta(j);
end
 


% =============================================================

end
