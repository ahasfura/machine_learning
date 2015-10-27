%% Problem 1

[test_error1, val_error1]=svm_test('stdev1', [1,2], 1)
[test_error2, val_error2]=svm_test('stdev2', [3,4], 1)
[test_error4, val_error4]=svm_test('stdev4', [5,6], 1)
[test_errorn, val_errorn]=svm_test('nonsep', [7,8], 1)


%%
tic
[w, S]=svm_test_kernel('stdev1', [1,2], .01, 10)
[w2, S2]=svm_test_kernel('stdev2', [3,4], .01,10)
[w4, S4]=svm_test_kernel('stdev4', [5,6], .01,10)
[wn, Sn]=svm_test_kernel('nonsep', [7,8], .01,10)
toc
%%
[w, S]=svm_test_kernel('stdev1', [1,2], .1, .1)
[w2, S2]=svm_test_kernel('stdev2', [3,4], .1,.1)
[w4, S4]=svm_test_kernel('stdev4', [5,6], .1,.1)
[wn, Sn]=svm_test_kernel('nonsep', [7,8], .1,.1)
%%

[w, S]=svm_test_kernel('stdev1', [1,2], 1, .1)
[w2, S2]=svm_test_kernel('stdev2', [3,4], 1,.1)
[w4, S4]=svm_test_kernel('stdev4', [5,6], 1,.1)
[wn, Sn]=svm_test_kernel('nonsep', [7,8], 1,.1)
%%
[w, S]=svm_test_kernel('stdev1', [1,2], 10, .1)
[w2, S2]=svm_test_kernel('stdev2', [3,4], 10,.1)
[w4, S4]=svm_test_kernel('stdev4', [5,6], 10,.1)
[wn, Sn]=svm_test_kernel('nonsep', [7,8], 10,.1)
%%

[w, S]=svm_test_kernel('stdev1', [1,2], 100, .1)
[w2, S2]=svm_test_kernel('stdev2', [3,4], 100,.1)
[w4, S4]=svm_test_kernel('stdev4', [5,6], 100,.1)
[wn, Sn]=svm_test_kernel('nonsep', [7,8], 100,.1)
