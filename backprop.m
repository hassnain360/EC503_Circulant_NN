function [gradW, gradx] = backprop(grada,W,x,p,q,k)
%L is the loss function
%a(i,l) is the lth output element in ai


gradW = zeros(p,q);
gradx = zeros(1,q);

for i = 1:p
    for j = 1:q
        gradW(i,j) = ifft(fft(grada).*fft(x(j)'));
        gradx(j) = gradx(j) + ifft(fft(grada(i).*fft(W(i,j))));
    end
end
end