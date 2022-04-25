%{ 
###########################################################################

                Dense Neural Network for MNIST Digit Recognition
    
                          EC503 Learning from Data
                           
                            Hasnain Abdur Rehman
                              Darya Smolina
            
                           
                          Boston University, 2022.

###########################################################################


                           Network Architecture 
            
                                4 Layers

                        [1] Input Layer    =   784
                        [2] Hidden Layer   =   1024
                        [3] Hidden Layer   =   512
                        [4] Output Layer   =   10



                         Weight Matrices Dimensions

                         W_12 = 784          x    1024
                         W_23 = (1024 + 1)   x    512
                         W_34 = (512 + 1)    x    10



                                Optimizer:
                  Stochastic Gradient Descent with Momemtum



%}

%%

% Load Data, Set Variables & Initialize Matrices

clc
clear all
format long g
load 'mnist'

lr = 0.05;                         % Learning Rate
batch_size = 1000;       
[d,m] = size(Xtr);
num_batches = m/batch_size;
epochs = 20;
total_loss = 0;
beta = 0.5;                        % Momemtum Hyperparameter 'B'
past_losses = [0 0 0];             % Vector Containing Past Batch Losses

W_12 = randn(784,1024)/10;         % Random Weight Initialization 
W_23 = randn(1024+1,512)/10;
W_34 = randn(512+1,10)/10;

mom_1 = zeros(size(W_12));         % Momemtum matrices
mom_2 = zeros(size(W_23));
mom_3 = zeros(size(W_34));

val_losses = [];

%%

% Train the NN using Back Propagation

for epoch = 1: epochs
    
    for j = 1:num_batches
       
       lr = lr*0.999;                          % Decrease Learning Rate each batch
       batch_loss = 0;
       
       for i = 1:batch_size
           
           idx = (j-1)*batch_size + i;
           
           % Perform forward pass
           act_1 = Xtr(:, idx);
           [act_2,act_3,act_4] = forwardprop(W_12,W_23,W_34,act_1);
            
           % Calculate Loss 
           y_t = onehotencode(ytr(idx)-1);
           loss = squared_loss(y_t, act_4);
           batch_loss = batch_loss + sum(loss);
        
           % Find Weight Updates through Back Propagation
           
           % Find del_w34 
           grad_w34 = softmax_layer_derivative(act_4,y_t);
           del_w34 = grad_w34' .*  repmat([1; act_3],1,10);
    
           % Find del_w23 
           grad_w23 = ReLU_derivative(act_3) .* sum(grad_w34' .* W_34(2:end,:), 2);
           del_w23 = grad_w23' .*  repmat([1; act_2],1,512);
    
           % Find del_w12
           grad_w12 = ReLU_derivative(act_2) .* sum(grad_w23' .* W_23(2:end,:), 2); 
           del_w12 = grad_w12' .* repmat (act_1, 1, 1024);
    
           % Update the weights using Momemtum
           W_34 = W_34 - lr*mom_3;
           W_23 = W_23 - lr*mom_2;
           W_12 = W_12 - lr*mom_1;

           mom_3 = beta*mom_3 + (1-beta)*del_w34;
           mom_2 = beta*mom_2 + (1-beta)*del_w23;
           mom_1 = beta*mom_1 + (1-beta)*del_w12;
          
       end
       
       total_loss = total_loss + batch_loss;
       past_losses = [batch_loss past_losses ];
       disp ('Last 3 Batch Losses: ');
       disp(past_losses(1:3));
   

       % Find Validation Error
       e = test_val(Xval,yval,W_12,W_23,W_34);
       val_losses = [val_losses e];
       disp('####################################### ' );
        
    end
end


    



