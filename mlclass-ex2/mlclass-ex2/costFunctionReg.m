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
% =============================================================

sum =0; 
for i=1:m
	h_theta = sigmoid(X(i, :)* theta);
	sum = sum + ((-y(i, 1) * log(h_theta)) - ((1-y(i,1))* log(1-h_theta))); 
end

% Calculate theta squares
theta_sq = 0; 
for i =2:length(theta)
	theta_sq = theta_sq + (theta(i, 1) * theta(i*1)); 
end
J = (sum / m) + ((lambda/(2*m))* theta_sq); 

% Calculate the gradient

for j=1:length(theta)
	sum = 0;
	reg = 0; 
	if j>=2
		reg = lambda * theta(j, 1);
	end

	for i =1:m 
		sum = sum + (sigmoid(X(i, :) * theta) - y(i, :)) * X(i, j); 
	end
	grad(j, 1) = ((sum+reg) / m); 
end


end
