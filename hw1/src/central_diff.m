function grad = central_diff(f, x, h)

h_vect = h * ones(size(x));
grad = (f(x + .5 * h_vect) - f(x - .5 * h_vect)) / h;

end