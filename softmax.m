function output = softmax(x)
    m = max(x);
    exp_x = exp(x -m);
    output= exp_x/sum(exp_x);
end