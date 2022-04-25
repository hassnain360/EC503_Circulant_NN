%{ 

Dense Neural Network for MNIST

Testing

%}

load mnist.mat          % load test dataset
%load 'mnist_weights.mat'
% test the dataset

[dim1, dim2] = size(Xte);

err = 0;
correct = 0;

for q = 1: dim2
    act_1 = Xte(:,q);
    [act_2,act_3,act_4] = forwardprop(W_12,W_23,W_34,act_1);
    y_hat = onehotencode(yte(q)-1);
    err = err + sum(squared_loss(y_hat,act_4));
    [mx, argmx] = max(act_4);
    if (argmx == yte(q))
        correct = correct+1;
    end
end
disp('Test Accuracy: ')
fprintf('%f %% \n\n\n',correct/dim2 * 100)



% report accuracy