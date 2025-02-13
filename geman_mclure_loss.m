function loss = geman_mclure_loss(x, c)
    % Geman-McClure loss function

    % Ensure c is provided, if not set a default value
    if nargin < 2
        c = 1;
    end
    
    % Compute the Geman-McClure loss
    loss = (x.^2) ./ (x.^2 + c^2);

    R = size(loss,2);
    loss = sum(loss)/R;
end
