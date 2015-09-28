function [analytical, numerical, diff] = comparator(f, x, h, g)

numerical = central_diff(f, x, h);
analytical = g(x);
diff = abs(numerical - analytical);

end