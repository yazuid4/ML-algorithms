function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

  U = [1 3 10 30];
  V = (1/100)*[1 3 10 30];
  Z = [U  V];
  R = [];

  for i=1:size(Z,2)
	for j=1:size(Z,2)
	   R = [R;Z(i) Z(j)];
     end
  end
  
  error = [];
Xval
yval
  
   for i=1:size(R,1)
    
p = svmPredict( svmTrain(X, y, R(i,1), @(Xval, yval) gaussianKernel(Xval, yval, R(i,2))), Xval);
    
error(i) = mean(double(p ~= yval));
   end
    
  [Q index] = min(error);
  
  C = R(index,1);
  sigma = R(index,2);

  C
  sigma



% =========================================================================

end
