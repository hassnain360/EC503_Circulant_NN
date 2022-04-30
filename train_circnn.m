%{ 
###########################################################################

        Dense Neural Network with Block Circulant Weight Matrices
                     for MNIST Classificaion
                       
                                   
                             EC 503 
                        Learning from Data
                                by
                         Francesco Orabona

            
                          
                       Hasnain Abdur Rehman
                          Darya Smolina
            
                           
                      Boston University, 2022.

###########################################################################


                        Network Architecture 
            
                             4 Layers

                     [1] Input Layer    =   785
                     [2] Hidden Layer   =   1025
                     [3] Hidden Layer   =   515
                     [4] Output Layer   =   10


                           Block Size = 5

###########################################################################




                   Weight Matrices before Compression
                              [m x n]
                             such that
                              a = W*x

                       W_12 = 1025         x      785
                       W_23 = 515          x   (1025 + 5)
                       W_34 = 10           x   ( 515 + 5)

                  
                    

                   Weight Matrices after Compression
                              [p x q]
                   where 
                   p = m
                   &
                   q = n / k,
                                    
                       W_12 = 205          x       785
                       W_23 = 103          x    (1025 + 5)
                       W_34 = 2            x     (515+5)

###########################################################################


                            Optimizer:
              Stochastic Gradient Descent with Momemtum


###########################################################################

%}

%%

% Load & Preprocess Data

clc
clear all
format long g
load 'mnist'

[d,m] = size(Xtr);
Xtr = [Xtr; zeros(1,m)];            % Pad X data with 0's.

[d,m] = size(Xte);
Xte = [Xte; zeros(1,m)];

[d,m] = size(Xval);
Xval = [Xval; zeros(1,m)];


%%
% Set Variables & Initialize Matrices

k = 5;                             % Block Size

lr = 0.001;                         % Learning Rate
batch_size = 1; 

epsilon = 0.1

input_size = 785;
hidden_1_size = 1025;
hidden_2_size = 515;
output_size = 10;

num_batches = m/batch_size;
epochs = 1;
total_loss = 0;
beta = 0.5;                        % Momemtum Hyperparameter 'B'
past_losses = [0 0 0];             % Vector Containing Past Batch Losses

W_12 = randn(205,785)/100;          % Weight Initialization 
W_23 = randn(103,1030)/100;         % for Block Circulant Matrices
W_34 = randn(2,520)/100;

del_w12 = zeros(size(W_12));       % Matrices to hold weight updates
del_w23 = zeros(size(W_23));
del_w34 = zeros(size(W_34));

batch_del_w12 = zeros(size(W_12));       % Matrices to hold weight updates
batch_del_w23 = zeros(size(W_23));
batch_del_w34 = zeros(size(W_34));

del_x2 = zeros(hidden_1_size+k,1);
del_x3 = zeros(hidden_2_size+k,1);


mom_1 = zeros(size(W_12));         % Momemtum matrices
mom_2 = zeros(size(W_23));
mom_3 = zeros(size(W_34));

val_losses = [];

%%

% Train the NN using Back Propagation

for epoch = 1: epochs
    
    for j = 1:num_batches
       
       % Decrease Learning Rate each batch
       lr = lr*0.999;                 
       batch_loss = 0;
       
       for i = 1:batch_size
           %fprintf('processing %d/%d\n',i,batch_size);
           idx = (j-1)*batch_size + i;
           
           % Perform forward pass
           act_1 = Xtr(:, idx);
           act_1 = act_1 + epsilon;
           act_1 = act_1 * 255;
           [act_2,act_3,act_4] = forwardprop_circnn(W_12,W_23,W_34,act_1,k);
            
           % Calculate Loss 
           y_t = onehotencode(ytr(idx)-1);
           loss = squared_loss(y_t, act_4);
           batch_loss = batch_loss + sum(loss);
        
           % Find Weight Updates through Back Propagation
           
           % Find del_w34 
           [d,m] = size(W_34);
           p = d;
           q = m/k;
           grad_w34 = softmax_layer_derivative(act_4,y_t);
           for ii = 1:p
               grad_w34_slice = grad_w34((ii-1)*k+1:ii*k,1);
               for jj = 1:q
                   % Calculate delta_w34
                   del_w34(ii,(jj-1)*k +1:jj*k) = ifft (fft(grad_w34_slice) .* fft(act_3((jj-1)*k+1:jj*k,1)));
                   % Calculate delta_x3
                   del_x3((jj-1)*k +1:jj*k,1) = del_x3((jj-1)*k +1:jj*k,1) + ifft(fft(grad_w34_slice) .* fft (W_34(ii,(jj-1)*k +1:jj*k)'));                              
               end
           end
           del_x3 = del_x3(k+1: end,:);



           % Find del_w23 
           [d,m] = size(W_23);
           p = d;
           q = m/k;
           grad_w23 = ReLU_derivative(del_x3);
           for ii = 1:p
               grad_w23_slice = grad_w23((ii-1)*k+1:ii*k,1);
               for jj = 1:q
                   % Calculate delta_w23
                   del_w23(ii,(jj-1)*k +1:jj*k) = ifft (fft(grad_w23_slice) .* fft(act_2((jj-1)*k+1:jj*k,1)));
                   % Calculate delta_x2
                   del_x2((jj-1)*k +1:jj*k,1) = del_x2((jj-1)*k +1:jj*k,1) + ifft(fft(grad_w23_slice) .* fft (W_23(ii,(jj-1)*k +1:jj*k)'));                              
               end
           end
           del_x2 = del_x2(k+1:end,:);



           % Find del_w12
           [d,m] = size(W_12);
           p = d;
           q = m/k;
           grad_w12 = ReLU_derivative(del_x2);
           for ii = 1:p
               grad_w12_slice = grad_w12((ii-1)*k+1:ii*k,1);
               for jj = 1:q
                   % Calculate delta_w34
                   del_w12(ii,(jj-1)*k +1:jj*k) = ifft (fft(grad_w12_slice) .* fft(act_1((jj-1)*k+1:jj*k,1)+0.0001));
                   % Calculate delta_x3
                   %del_x1((jj-1)*k +1:jj*k,1) = del_x3((jj-1)*k +1:jj*k,1) + ifft(fft(grad_w34_slice) .* fft (W_34(ii,(jj-1)*k +1:jj*k)));                              
               end
           end

       batch_del_w34 = batch_del_w34 + del_w34;
       batch_del_w23 = batch_del_w23 + del_w23;
       batch_del_w12 = batch_del_w12 + del_w12;


        del_x2 = zeros(hidden_1_size+k,1);
        del_x3 = zeros(hidden_2_size+k,1);
           
          
       end


        % Update the weights using Momemtum
        
        mom_3 = beta*mom_3 + (1-beta)*batch_del_w34;
        mom_2 = beta*mom_2 + (1-beta)*batch_del_w23;
        mom_1 = beta*mom_1 + (1-beta)*batch_del_w12;
        
        W_34 = W_34 - lr*mom_3;
        W_23 = W_23 - lr*mom_2;
        W_12 = W_12 - lr*mom_1;
    
        
     

        del_w12 = zeros(size(W_12));       % Matrices to hold weight updates
        del_w23 = zeros(size(W_23));
        del_w34 = zeros(size(W_34));


        batch_del_w12 = zeros(size(W_12));       % Matrices to hold weight updates
        batch_del_w23 = zeros(size(W_23));
        batch_del_w34 = zeros(size(W_34));

       
       total_loss = total_loss + batch_loss;
       past_losses = [batch_loss past_losses ];
       disp ('Last 3 Batch Losses: ');
       disp(past_losses(1:3));
   

       % Find Validation Error
       %e = test_val_circnn(Xval,yval,W_12,W_23,W_34,k);
       %val_losses = [val_losses e];
       disp('####################################### ' );
        
    end
end


    



