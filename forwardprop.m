function [a,b] = forwardprop(W,x,p,q,k) %confused as to why this needs k as an input
a = zeros(p,q);
b = zeros(p,q);
%P = m/k
%Q = n/k
%x is 1:q
%K = block size(size of each sub matrix)
%W = m by n weight matrix that is partitioned into square submatrix (each submatrix is a circulant matrix)
%So W = Wij where i is 1:p and j is 1:q
%A (o/p) = p rows by q columns
%A = Wx = col vector a1…ap 
for i = 1:p 
   for j = 1:q 
       a(:,i) = a(:,i) + W(i,j)*x(j);
   end
end
%end
for i = 1:p
    for j = 1:q
        b(:,i) = b(:,i) + ifft(fft(W(i,j)).*fft(x(j)));
    end
end
end