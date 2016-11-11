function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
m= size(X,1);
K = size(centroids, 1);

% You need to return the following variables correctly.
#idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%idx(i) corresponds to cˆ(i), the index of the centroid assigned to example i
%each row of X is a single example

d=zeros(K, m);

for i = 1 : m
  #example i and all centroids
  diff = X(i, :)-centroids ;
  for j =1 : K
    d(j, i)=sum(diff(j, :).^2);
    endfor
  endfor
  
  [this_min, imin]=min(d); 
  idx=imin';
  
% =============================================================

end

