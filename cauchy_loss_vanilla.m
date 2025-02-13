function loss = cauchy_loss_vanilla(x)
%Cauchy_loss_vanilla  Compute the Cauchy loss for input residuals.

%   The computed loss is then averaged across the third dimension.
%
%   Input:
%       x - Residual values (array).
%
%   Output:
%       loss - Averaged loss value.
    
    % Compute the element-wise Cauchy loss
    loss = log(0.5 * (abs(x)).^2 + 1);
    
    % Average the loss across the third dimension
    % If x is not a 3-D array, size(loss,3) returns 1, and the operation is valid.
    R = size(loss, 2);
    loss = sum(loss) / R;
end
