function err = test_val(Xval,yval,W_12,W_23,W_34);

[dim1, dim2] = size(Xval);

err = 0;
correct = 0;

for q = 1: dim2
    act_1 = Xval(:,q);
    [act_2,act_3,act_4] = forwardprop(W_12,W_23,W_34,act_1);
    y_hat = onehotencode(yval(q)-1);
    err = err + sum(squared_loss(y_hat,act_4));
    [mx, argmx] = max(act_4);
    if (argmx == yval(q))
        correct = correct+1;
    end
end
disp('Validation Accuracy: ')
disp(correct/dim2 * 100)

end