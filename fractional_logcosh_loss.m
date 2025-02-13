function loss = fractional_logcosh_loss(x, alpha, n_memory, h)
% FRACTIONAL_LOGCOSH_LOSS Compute the fractional derivative of the logcosh loss.
%
%   loss = fractional_logcosh_loss(x, alpha, n_memory, h) computes a fractional
%   derivative variant of the logcosh loss function using a series expansion
%   based on the Grünwald–Letnikov definition:
%
%   Inputs:
%       x         - Residual values (array).
%       alpha     - Order of the fractional derivative (scalar).
%       n_memory  - Number of terms in the series expansion (non-negative integer).
%       h         - Step size for the finite difference approximation (scalar).
%
%   Output:
%       loss      - Averaged fractional logcosh loss (scalar).
%
    
    % Normalization factor (set to 1; modify if needed)
    normalization_factor = 1;
    
    % Initialize the series sum
    sum_value = 0;
    
    % Series expansion: compute the fractional derivative using the Grünwald–Letnikov approach.
    % f(u) = log(cosh(|u|)) and u = x - n*h.
    for n = 0:n_memory
        % Evaluate f(x - n*h)
        f_val = log(cosh(abs(x - n * h)));
        
        % Compute the fractional binomial coefficient:
        %   binom = Gamma(a+1) / (Gamma(n+1) * Gamma(a - n + 1))
        binom = gamma_frac(alpha + 1, 1) / (gamma_frac(n + 1, 1) * gamma_frac(alpha - n + 1, 1));
        
        % Compute and accumulate the current term
        term = (-1)^n * binom * f_val / normalization_factor;
        sum_value = sum_value + term;
    end
    
    % Scale the series sum by h^alpha to approximate the fractional derivative
    loss = sum_value / h^alpha;
    
    % For integer values of alpha, compute the standard derivative of f(x)
    if alpha == floor(alpha)
        syms t
        f = log(cosh(abs(t)));
        for n = 1:alpha
            f = diff(f, t);
        end
        loss = double(subs(f, t, x));
    end
    
    % Average the loss over all elements
    loss = mean(loss(:));
end
