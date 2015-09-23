function [min_x, min_val] = grad_descent(f, g, x_start, step_size, precision)
% implementation of gradient descent    

x_new = x_start;
x_old = x_new - 2 * precision; % make sure it doesn't converge on first step

while abs(x_new - x_old) < precision
    x_old = x_new;
    x_new = x_old - step_size * g(x_old);
end

min_x = x_new;
min_val = f(min_x);

end