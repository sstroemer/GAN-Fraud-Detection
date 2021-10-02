import pandas as pd
from time import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import torch
from gan import Discriminator, Generator, VanillaGAN


# load 10y government bond data
raw_de = pd.read_csv('data/IRLTLT01DEM156N.csv').set_index('DATE').round(2)
raw_ie = pd.read_csv('data/IRLTLT01IEM156N.csv').set_index('DATE').round(2)
data = raw_de.join(raw_ie)
data.rename(columns={'IRLTLT01DEM156N': 'DE', 'IRLTLT01IEM156N': 'IE'}, inplace=True)

# normalize both
mean_DE = data['DE'].mean()
std_DE = data['DE'].std()
data['DE_norm'] = (data['DE'] - mean_DE) / std_DE
mean_IE = data['IE'].mean()
std_IE = data['IE'].std()
data['IE_norm'] = (data['IE'] - mean_IE) / std_IE

# plot the data to compare with original toy example
#! fig = px.scatter(x=data.DE, y=data.IE)
#! fig.show()


# noise and data function
noise_fn = lambda x: torch.rand((x, 2), device='cpu')
data_fn = lambda x: torch.tensor(data.iloc[torch.randint(len(data), (x,), device='cpu')][['DE_norm', 'IE_norm']].values).float()

# generate a fixed noise set to plot afterwards
test_noise = noise_fn(2000)

# prep
s = 100
heatmap_input = torch.Tensor([[
    i * (data.DE_norm.max() - data.DE_norm.min()) / s + data.DE_norm.min(),
    j * (data.IE_norm.max() - data.IE_norm.min()) / s + data.IE_norm.min()
] for i in range(s + 1) for j in range(s + 1)])

fake_fn = lambda x: heatmap_input[torch.randint(heatmap_input.shape[0], (x,), device='cpu')]


epochs = 15001
batches = 10
generator = Generator(2, [32, 32, 16, 16, 2])
discriminator = Discriminator(2, [32, 32, 16, 16, 1])
gan = VanillaGAN(generator, discriminator, noise_fn, data_fn, fake_fn, device='cpu')
loss_g, loss_d_real, loss_d_fake = [], [], []
start = time()

scatter_real = go.Scatter(x=data.DE, y=data.IE, name='REAL', mode='markers')



for epoch in range(epochs):
    if (epoch % 200) == 199:
        gan.nof_fake = 32
    else:
        gan.nof_fake = 0

    loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
    for batch in range(batches):
        lg_, (ldr_, ldf_) = gan.train_step()
        loss_g_running += lg_
        loss_d_real_running += ldr_
        loss_d_fake_running += ldf_
    loss_g.append(loss_g_running / batches)
    loss_d_real.append(loss_d_real_running / batches)
    loss_d_fake.append(loss_d_fake_running / batches)

    if (epoch % 100) == 0:
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" G={loss_g[-1]:.3f},"
              f" Dr={loss_d_real[-1]:.3f},"
              f" Df={loss_d_fake[-1]:.3f}")

    if (epoch % 500) == 0:
        test_sample = gan.generate_samples(latent_vec=test_noise)

        scatter_gen = go.Scatter(x=test_sample[:, 0]*std_DE+mean_DE, y=test_sample[:, 1]*std_IE+mean_IE, name='GEN', mode='markers')
        fig = make_subplots()
        fig.add_trace(scatter_real)
        fig.add_trace(scatter_gen)
        fig.show()



heatmap_value = torch.transpose(torch.reshape(discriminator.forward(heatmap_input).detach(), (s + 1, s + 1)), 0, 1)

fig = make_subplots()
fig.add_trace(go.Heatmap(z=heatmap_value, x=heatmap_input[0:heatmap_input.shape[0]:(s+1), 0]*std_DE+mean_DE, y=heatmap_input[0:(s+1), 1]*std_IE+mean_IE, colorscale='rdylgn'))
fig.add_trace(go.Scatter(x=data.DE, y=data.IE, mode='markers', marker={'color': 'black'}))
fig.show()
