a = randn(205,785)/100;
b = randn(103,1030)/100;
c = randn(2,520)/100;


input = randn(785,1);


[a2,a3,a4] = forwardprop_circnn(a,b,c,input,5);