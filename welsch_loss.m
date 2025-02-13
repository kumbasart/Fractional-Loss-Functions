function loss = welsch_loss(x, c)
    % Welsch loss function

    if nargin < 2
        c = 1;
    end
    
    % Compute the Welsch loss
    loss = 1 - exp(-(x.^2) / (2 * c^2));

    R = size(loss,2);
    loss = sum(loss)/R;
end