function J = costFunction(X, centroids)
% Set K
m= size(X,1);
K = size(centroids, 1);

J=0;
for i = 1 : m
  #example i and all centroids
  diff = X(i, :)-centroids ;
  for j =1 : K
    J=J+sum(diff(j, :).^2);
    endfor
  endfor

  J=J/m;
  end

