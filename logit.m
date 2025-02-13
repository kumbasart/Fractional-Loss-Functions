function result = logit(y)
    % The inverse of the sigmoid function.

    result = -log(1.0 ./ y - 1.0);
end
