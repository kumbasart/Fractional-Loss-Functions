function [loss, gradients, state] = modelLoss(net1, X, T, Learnable_parameters, iteration_no, bounds, lossFunctionName)
% modelLoss  Compute the loss and gradients for the neural network.
%
%   [loss, gradients, state] = modelLoss(net1, X, T, Learnable_parameters, iteration_no, bounds, lossFunctionName)
%
%   Inputs:
%       net1               - dlnetwork object representing the neural network.
%       X                  - Input data (dlarray).
%       T                  - Target data (dlarray).
%       Learnable_parameters - Structure containing learnable parameters (e.g., alpha and net1).
%       iteration_no       - Current iteration number (unused here but kept for compatibility).
%       bounds             - Two-element vector [lower, upper] for affine sigmoid transformation of alpha.
%       lossFunctionName   - String specifying the loss function to be used.
%
%   Outputs:
%       loss      - Computed loss value.
%       gradients - Gradients of the loss with respect to the learnable parameters.
%       state     - Updated state of the network after the forward pass.
%
%   Available loss functions (ensure the corresponding files are in the path):
%       'cauchy_loss_vanilla'
%       'fractional_cauchy_loss'
%       'fractional_L2_loss'
%       'fractional_logcosh_loss'
%       'geman_mclure_loss'
%       'L2_loss_vanilla'
%       'logcosh_loss_vanilla'
%       'welsch_loss'

    %% Synchronize Learnable Parameters
    % Update the network learnables from the current Learnable_parameters
    net1.Learnables = Learnable_parameters.net1;
    Learnable_parameters.net1 = net1.Learnables;

    %% Forward Pass
    % Compute the network output and updated state.
    [Y, state] = forward(net1, X);

    % Compute the residual (error) between predictions and targets.
    errorSignal = Y - T;

    %% Loss Function Parameters
    % Parameters for loss functions that require memory and step size.
    n_memory = 5;
    h_step_size = 0.1;

    % Transform the learnable alpha parameter using an affine sigmoid.
    alpha = affine_sigmoid(Learnable_parameters.alpha, bounds(1), bounds(2));

    %% Loss Computation: Select Loss Function Based on Input String
    switch lossFunctionName
        case 'cauchy_loss_vanilla'
            loss = cauchy_loss_vanilla(errorSignal);
        case 'fractional_cauchy_loss'
            loss = fractional_cauchy_loss(errorSignal, alpha, n_memory, h_step_size);
        case 'fractional_L2_loss'
            loss = fractional_L2_loss(errorSignal, alpha);
        case 'fractional_logcosh_loss'
            loss = fractional_logcosh_loss(errorSignal, alpha, n_memory, h_step_size);
        case 'geman_mclure_loss'
            loss = geman_mclure_loss(errorSignal);
        case 'L2_loss_vanilla'
            loss = L2_loss_vanilla(errorSignal);
        case 'logcosh_loss_vanilla'
            loss = logcosh_loss_vanilla(errorSignal);
        case 'welsch_loss'
            loss = welsch_loss(errorSignal);
        otherwise
            error('Unknown loss function: %s', lossFunctionName);
    end

    %% Gradient Computation
    % Compute gradients with respect to the learnable parameters.
    gradients = dlgradient(loss, Learnable_parameters);

end
