function loss = fractional_L2_loss(x, alpha)
% FRACTIONAL_L2_LOSS Compute the fractional L2 loss.
%   Inputs:
%       x     - Residual values (array).
%       alpha - Order of the derivative (scalar).
%
%   Output:
%       loss  - Averaged fractional L2 loss (scalar).
%
    % Set the scaling constant
    c = 0.1;
    
    if alpha ~= floor(alpha)
        % For non-integer alpha, compute the fractional L2 loss
        normalization_factor = 1;  % Can be adjusted if needed
        
        % Compute fractional L2 loss:
        %   g(x) = (|x/c|^(2 - alpha)) / Gamma(3 - alpha)
        g_x = (abs(x / c).^(2 - alpha)) / gamma_frac(3 - alpha, 1);
        loss = g_x / normalization_factor;
    else
        % For integer alpha, compute the standard derivative of f(t)=0.5*t^2.
        % For example, if alpha = 1, then f'(t) = t, and if alpha = 2, f''(t) = 1.
        syms t
        f = 0.5 * t.^2;
        for n = 1:alpha
            f = diff(f, t);
        end
        loss = double(subs(f, t, x));
    end
    
    % Average the loss along the second dimension
    R = size(loss, 2);
    loss = sum(loss, 2) / R;
end
