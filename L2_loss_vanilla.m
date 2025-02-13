function loss = L2_loss_vanilla(x)
c=0.1;
loss = 0.5*(x./c).^2;

R = size(loss,2);
loss = sum(loss)/R;
