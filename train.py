import torch.optim as optim
from utils import *
from nets import MRspecNET
from parameter_values import *
import matplotlib.pyplot as plt

#-------------------------------------------------
# Load Basis Sets:
#-------------------------------------------------
mmbg_path  = 'data/BasisSet/MMBG/MMBG_050_woCrCH2.mat'
metab_path = 'data/BasisSet/dSTEAM_D050'

metab_basis = Metab_basis(metab_path, kwargs_BS)
mmbg_basis  = MMBG_basis(mmbg_path, kwargs_MM)

#-------------------------------------------------
# Set up model and simulator for training:
#-------------------------------------------------
simulator   = Simulator(metab_basis, mmbg_basis)

ppmAx, fAx, wCenter, fL = build_ppmAx(bw, noSmp)
device     = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_built() else torch.device('cpu')
epochs     = 5000
lr         = 5e-5
batch_size = 64
print('device: ', device)
model = MRspecNET(kernel_size=32, n_channels=64, n_layers=6).to(device)

n_params = sum(p.numel() for p in model.parameters())
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
losses = []
best_loss = 1e-3 # model is automatically saved if loss<best_loss

#-------------------------------------------------
# Main Training Loop:
#-------------------------------------------------
for epoch in range(epochs+1):
    signal_batch, noisy_batch = make_batch(batch_size, simulator, device, include_mmbg=True,
                                              restrict_range=(1600,2400), normalization='max_1', **kwargs_BS)
    pred = model(noisy_batch)
    loss = loss_fn(pred, signal_batch)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100==0 and epoch>0:
        print_info(losses,lr,epoch,epochs,n_params)
        plot_losses(losses, y_mode='log')
    current_loss = np.mean(losses[-100:])
    if current_loss<best_loss:
        timer = 0
        best_loss = current_loss
        print('best loss: ', "{:e}".format(best_loss))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'batch_size': batch_size,
            'learning_rate': lr,
            'kwargs_BS': kwargs_BS,
            'kwargs_MM': kwargs_MM
        }, model.name)
plt.show()





















