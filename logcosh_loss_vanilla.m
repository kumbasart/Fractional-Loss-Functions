function loss = logcosh_loss_vanilla(x)

loss = (log(cosh(abs(x))));

R = size(loss,2);
loss = sum(loss)/R;
