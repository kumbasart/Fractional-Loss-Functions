function out=gamma_frac(z,t)

    gamma_euler = 0.57721;
    k = 1:10000;

    % Element-wise operations for the product computation
    prod_array = ((1 + (z./k)).^-1) .* exp(z./k);

    out = (exp(-gamma_euler*z)/z) * prod(prod_array);

end