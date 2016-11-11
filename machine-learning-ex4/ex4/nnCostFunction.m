function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
#J = 0;
#Theta1_grad = zeros(size(Theta1));
#Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Add ones to the X data matrix
a1 = [ones(m,1) X];

# a2=g(X*Theta1)
z2 = a1*Theta1';
a2 = sigmoid(z2);#one vector of hidden_layer_size elements for each training set: matrix (m, hidden_layer_size)
a2 = [ones(m, 1) a2];

# a3=g(a2*Theta2)=h(theta)
a3 = sigmoid(a2*Theta2'); #one vector of num_label elements for each training set: matrix (m, num_label)
# no ones added at the last step

#each y_i is transformed into a vector of dimension num_labels
Y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels); #copied... genius!

#Y=(m, num_labels), a3=(m, num_labels)
# element wise, because I want each a3(i) to multiply its correspondent Y(i) 
J = (-Y.*log(a3)) - ((1-Y).*log(1-a3)); # matrix(num_labels, m)
# and nothing else
J = sum(sum(J));
J = J/m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

# prediction
#[maxprob, imaxprob]=max(a3, [], 2);
#p=imaxprob;
#P= repmat([1:num_labels], m, 1) == repmat(p, 1, num_labels);

# delta 3
d3 = a3 - Y;
# delta 2: remove firts column in Theta2 (bias) and use z2 
#(because there is no added bias unit there
Theta2_red=Theta2(:,2:end);
Theta1_red=Theta1(:,2:end);
d2 = ((d3*Theta2_red).*sigmoidGradient(z2));

grad_Theta1 = 1/m * d2' * a1;;
grad_Theta2 = 1/m * d3' * a2;;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%add regularization (just modify the elements that are gonna be different)
#avoid innecessary manipulation of matrices:

grad_Theta1(:, 2:end) = grad_Theta1(:, 2:end) + lambda/m*Theta1_red;
grad_Theta2(:, 2:end) = grad_Theta2(:, 2:end) + lambda/m*Theta2_red;
regJ=lambda/(2*m)*(sum(sum(Theta1_red.^2))+sum(sum(Theta2_red.^2)));
J=J+regJ;

#{  
#not fully vectorized version: much slower
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for i = 1:m
  this_a1=a1(i, :);
  this_a2=a2(i, :);
  this_d3=d3(i, :);
  #delta 2
  this_d2 = this_d3*Theta2 .* this_a2.*(1-this_a2);

  #Delta1: matrix of partial derivatives wrt Theta1 
  #Delta1: d2 is (1, hidden_layer_size+1) and a1 is (1,input_layer_size+1)
  #Delta1 has to have the same dim as Theta1, so (hidden_layer_size, input_layer_size+1)
  Delta1 = Delta1 + this_d2'(2:end)* this_a1;
  
  #Delta2: matrix of partial derivatives wrt Theta2 
  #d3 has not bias term
  Delta2 = Delta2 + this_d3'* this_a2;
  endfor
  
  grad_Theta1 = (1 / m) * Delta1 ;
  grad_Theta2 = (1 / m) * Delta2 ;

  #take off the bias term in Theta (because we don't need it for the gradients)
  Theta1_red=Theta1(:, 2:end);
  Theta2_red=Theta2(:, 2:end);

  regJ=lambda/(2*m)*(sum(sum(Theta1_red.*Theta1_red))+sum(sum(Theta2_red.*Theta2_red)));

  regGrad1=lambda/m*Theta1_red;
  regGrad1 = [zeros(hidden_layer_size, 1) regGrad1];

  regGrad2=lambda/m*Theta2_red;
  regGrad2 = [zeros(num_labels, 1) regGrad2];

  J=J+regJ;
  grad_Theta1 = grad_Theta1 + regGrad1;
  grad_Theta2 = grad_Theta2 + regGrad2;
#}

% Unroll gradients
grad = [grad_Theta1(:) ; grad_Theta2(:)];
% -------------------------------------------------------------

% =========================================================================

end
