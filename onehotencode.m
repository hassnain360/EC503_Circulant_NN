function output = onehotencode(y)

output = zeros(1,10);
output(y+1) = 1;

output=output';
end