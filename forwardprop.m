function [activations_2,activations_3,activations_4] = forwardprop(W_12,W_23,W_34,activations_1)

activations_2 = ReLU(W_12' * activations_1 );
activations_3 = ReLU(W_23' * [1; activations_2] );
activations_4 = softmax(W_34' * [1; activations_3] );

end


%{
a = zeros(k*p);
b = zeros(k*p);
%P = m/k
%Q = n/k
%x is 1:q
%K = block size(size of each sub matrix)
%W = m by n weight matrix that is partitioned into square submatrix (each submatrix is a circulant matrix)
%So W = Wij where i is 1:p and j is 1:q
%A (o/p) = p rows by q columns
%A = Wx = col vector a1…ap 
%for i = 1:p 
 %  for j = 1:q 
  %     a(i:(i+k)) = a(i:(i+k)) + W(i,j)*x(j);
  % end
%end
%end
%for i = 1:p
 %   for j = 1:q
  %      b(i) = b(i) + ifft(fft(W(i,j)).*fft(x(j)));
   % end
%end
%}



