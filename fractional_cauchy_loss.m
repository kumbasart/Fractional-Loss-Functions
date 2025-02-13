function loss = fractional_cauchy_loss(x, alpha, n_memory, h)

a = alpha;

c=1;

sum_value = 0;

for n = 0:n_memory

    log10_x = (log(0.5*(abs(x./c)-n*h).^2+1));

    term = (-1)^n * gamma_frac(a+1,1) * log10_x / ...
           (gamma_frac(n+1,1) * gamma_frac(1 - n + a, 1));
    sum_value = sum_value + term;
end

% Calculate the final value
loss = sum_value / h^a;

R = size(loss,2);
loss = sum(loss)/R;
