function [SSE]= test_sum_of_squares_error(Yp, Y)


SSE= (Yp-Y)'* (Y-Y); 

end 