function errs = softmax_layer_derivative(y_hat,y_t)

[m,n] = size(y_hat);

errs = zeros(m,n);

for i = 1 :m
    temp = 0;
    for j = 1:m
        if (j~=i)
            temp = temp + (- 1* y_hat(i) * y_hat(j)  )      *     y_hat(j) * (y_hat(j) - y_t(j)) ;
        else
            temp = temp + ( (y_hat(i) * (1 - y_hat(i))) )   *     y_hat(j) * (y_hat(j) - y_t(j)) ;

        end
    end
    errs(i)  = temp;
end


%output = values*(1-values);

end
