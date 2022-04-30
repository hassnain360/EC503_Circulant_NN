function [act_2, act_3, act_4] = forwardprop_circnn(W_12,W_23,W_34,act_1,k)
% Forward Propagation 


% Activations for Hidden Layer 1
[d,m] = size(W_12);
p = d;
q = m/k;
act_2 = zeros(p*k,1);
for i = 1 : p
    temp = zeros(k,1);
    for j = 1 : q
        temp = temp + ifft(fft(W_12(i,(j-1)*k+1: j*k)') .* fft(act_1((j-1)*k +1: j*k,:)));
    end
    act_2((k*(i-1))+1: i*k,:) = temp;
end
% Add 5 Bias Neurons
act_2 = [1; 1; 1; 1; 1; ReLU(act_2)];

% Activations for Hidden Layer 2
[d,m] = size(W_23);
p = d;
q = m/k;
act_3 = zeros(p*k,1);
for i = 1 : p
    temp = zeros(k,1);
    for j = 1 : q
        temp = temp + ifft(fft(W_23(i,(j-1)*k+1: j*k)') .* fft(act_2((j-1)*k +1: j*k,:)));
    end
    act_3((k*(i-1))+1: i*k,:) = temp;
end
% Add 5 Bias Neurons
act_3 = [1; 1; 1; 1; 1; ReLU(act_3)];


% Activations for Hidden Layer 1
[d,m] = size(W_34);
p = d;
q = m/k;
act_4 = zeros(p*k,1);

for i = 1 : p
    temp = zeros(k,1);
    for j = 1 : q
        temp = temp + ifft(fft(W_34(i,(j-1)*k+1: j*k)') .* fft(act_3((j-1)*k +1: j*k,:)));
    end
    act_4((k*(i-1))+1: i*k,:) = temp;
end
act_4 = softmax(act_4);

end

