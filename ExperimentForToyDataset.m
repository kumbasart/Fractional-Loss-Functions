%% Neural Network Regression with Selectable Loss Function
% This script demonstrates neural network regression on a toy dataset
% with the ability to switch between various loss functions. The loss 
% function is selected via a string variable.
%
% Loss function options (choose one):
%   'cauchy_loss_vanilla'
%   'fractional_cauchy_loss'
%   'fractional_L2_loss'
%   'fractional_logcosh_loss'
%   'gamma_frac'
%   'geman_mclure_loss'
%   'inv_affine_sigmoid'
%   'L2_loss_vanilla'
%   'logcosh_loss_vanilla'
%   'welsch_loss'
%
% Ensure that each of these functions exists in your path along with 
% modelLoss.m (which should be modified to accept lossFunctionName).
%
% Author: Mert Can Kurucu
%% Setup Environment
clear; clc; close all;
rng(1); % For reproducibility

%% Animation and File Settings
filename = 'training_animation.gif';
frameRate = 60;

%% Select Loss Function
% Choose one of the available loss functions by setting the following string.
lossFunctionName = 'fractional_L2_loss'; % e.g., 'fractional_L2_loss'

%% Generate Toy Dataset
n = 50; 
scale_true = 0.7;
shift_true = 0.15;
x = rand(1, n);
y = scale_true * x + shift_true;
y = y + 0.025 * randn(1, n);

% Flip some labels to introduce outliers
flip_mask = rand(1, n) > 0.9;
y(flip_mask) = 0.05 + 0.4 * (1 - sign(y(flip_mask) - 0.5));

% Convert to column vectors
x = x';
y = y';

%% Define Neural Network Architecture
numHiddenUnits = 1;
layers = [
    featureInputLayer(1,'Name','input')
    fullyConnectedLayer(1,'Name','fc')
];

%% Plot Setup for Animation
figure; 
scatter(x, y, 'filled'); 
hold on;
h = animatedline('Color', 'r', 'LineWidth', 2); % For NN predictions

%% Training Options and Hyperparameters
numEpochs = 1000;
initialLearnRate = 0.01;
miniBatchSize = n;
totalIterations = numEpochs * ceil(numel(x)/miniBatchSize);

% Create datastore and minibatch queue
trainDs = arrayDatastore([x,y]);
mbqTrain = minibatchqueue(trainDs, ...
    'MiniBatchSize', miniBatchSize, ...
    'MiniBatchFormat', ["BC"], ...
    'OutputEnvironment', 'cpu');

% Initialize dlnetwork
net1 = dlnetwork(layers);

% Adam optimizer variables
averageGrad = [];
averageSqGrad = [];
iteration = 0;

% Regularization and momentum parameters (if needed)
l2Regularization = 0;
momentum = 0.95;

% Initialize learnable parameter for fractional order (alpha)
alpha_init = dlarray(single(0.01));
lo_a = 0; hi_a = 1;
bounds = [lo_a, hi_a];
Learnable_parameters.alpha = inv_affine_sigmoid(alpha_init, lo_a, hi_a);
Learnable_parameters.net1 = net1.Learnables;

% Preallocate arrays for monitoring
alpha_values   = zeros(totalIterations, 1);
loss_values    = zeros(totalIterations, 1);
runningLoss_values = zeros(numEpochs, 1);

%% Training Loop
tic;
for epoch = 1:numEpochs
    runningLossSum = 0;
    lossCount = 0;
    
    % Shuffle training data at each epoch
    shuffle(mbqTrain);
    
    while hasdata(mbqTrain)
        iteration = iteration + 1;
        X = next(mbqTrain);
        
        % Compute loss and gradients.
        [loss, gradients, state] = dlfeval(@modelLoss, net1, X(1,:), X(2,:), ...
            Learnable_parameters, iteration, bounds, lossFunctionName);
        net1.State = state;
        
        % Update learnable parameters using Adam update rule
        [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(...
            Learnable_parameters, gradients, averageGrad, averageSqGrad, iteration, initialLearnRate);
        
        % Update alpha using the affine sigmoid transformation
        alpha_values(iteration) = extractdata(affine_sigmoid(Learnable_parameters.alpha, lo_a, hi_a));
        
        % Track loss for the current mini-batch
        runningLossSum = runningLossSum + double(loss);
        lossCount = lossCount + 1;
        loss_values(iteration) = double(loss);
    end
    
    % Compute and store average loss for the epoch
    averageLoss = runningLossSum / lossCount;
    runningLoss_values(epoch) = averageLoss;
    
    % Update network learnable parameters
    net1.Learnables = Learnable_parameters.net1;
    Learnable_parameters.net1 = net1.Learnables;
    
    % Update predictions for animation
    all_X = dlarray(x, "BC");
    predictions = predict(net1, all_X);
    clearpoints(h);
    addpoints(h, all_X, predictions);
    title(['Epoch: ' num2str(epoch)]);
    drawnow limitrate;
    
    % Capture and write animation frame to GIF
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if epoch == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 1/frameRate);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1/frameRate);
    end
end
toc;

%% Post-Training: Plot Results
% Plot data and final neural network regression line
legend('Data', 'Neural Network Regression', 'Location', 'best');
xlabel('x'); ylabel('y');
title('Neural Network Regression with Different Loss Functions');
set(gca, 'FontSize', 18);
hold off;

% Plot the learned fractional derivative order (alpha) over iterations
figure;
plot(0:totalIterations-1, alpha_values, 'LineWidth', 2);
title("Fractional Derivative Order Learning");
xlabel("Iteration"); ylabel("Alpha");
grid on;

% Plot the running loss per epoch
figure;
plot(0:numEpochs-1, runningLoss_values, 'b', 'LineWidth', 2);
grid on;
title("Running Loss per Epoch");
xlabel("Epoch"); ylabel("Loss");
legend('Running Loss');

%% Evaluation Metrics
all_X = dlarray(x, "BC");
predictions = predict(net1, all_X);

true_values = y;
predicted_values = gather(predictions); % Convert dlarray to numeric array if needed

% Calculate Root Mean Squared Error (RMSE)
RMSE = sqrt(mean((true_values - predicted_values).^2));
fprintf('Root Mean Squared Error: %.3f\n', RMSE);

% Calculate Mean Absolute Error (MAE)
MAE = mean(abs(true_values - predicted_values));
fprintf('Mean Absolute Error: %.3f\n', MAE);
