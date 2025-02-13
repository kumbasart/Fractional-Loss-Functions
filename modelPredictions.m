function predictions = modelPredictions(net,mbq)

predictions = [];

while hasdata(mbq)
    X = next(mbq);

    % Make predictions using the model function.
    Y = predict(net,X);
    % Extract the underlying data from the dlarray, then transpose.
    YData = extractdata(Y);
    YData = YData';
    
    % Determine predicted classes.
    predictions = [predictions; YData];
end

end