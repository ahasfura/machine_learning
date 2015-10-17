%% Problem 1

[test_error1, val_error1]=lr_test('stdev1', [1,2], 0)
[test_error2, val_error2]=lr_test('stdev2', [3,4], 0)
[test_error4, val_error4]=lr_test('stdev4', [5,6], 0)
[test_errorn, val_errorn]=lr_test('nonsep', [7,8], 0)


%%
[test_error1, val_error1]=lr_test('stdev1', [1,2], 1)
[test_error2, val_error2]=lr_test('stdev2', [3,4], 1)
[test_error4, val_error4]=lr_test('stdev4', [5,6], 1)
[test_errorn, val_errorn]=lr_test('nonsep', [7,8], 1)
%%
[test_error1, val_error1]=lr_test('stdev1', [1,2], 100)
[test_error2, val_error2]=lr_test('stdev2', [3,4], 100)
[test_error4, val_error4]=lr_test('stdev4', [5,6], 100)
[test_errorn, val_errorn]=lr_test('nonsep', [7,8], 100)

%%

[test_error1, val_error1]=lr_test('stdev1', [1,2], 10)
[test_error2, val_error2]=lr_test('stdev2', [3,4], 10)
[test_error4, val_error4]=lr_test('stdev4', [5,6], 10)
[test_errorn, val_errorn]=lr_test('nonsep', [7,8], 10)

%%
%%
[test_error1, val_error1]=lr_test('stdev1', [1,2], 1000)
[test_error2, val_error2]=lr_test('stdev2', [3,4], 1000)
[test_error4, val_error4]=lr_test('stdev4', [5,6], 1000)
[test_errorn, val_errorn]=lr_test('nonsep', [7,8], 1000)


%%

[test_error1, val_error1]=lr_test('stdev1', [1,2], 2000)
[test_error2, val_error2]=lr_test('stdev2', [3,4], 2000)
[test_error4, val_error4]=lr_test('stdev4', [5,6], 2000)
[test_errorn, val_errorn]=lr_test('nonsep', [7,8], 2000)