import numpy as np

def rate_encoding(signal, timesteps=50, max_rate=20):
    spikes = np.zeros((timesteps, len(signal)))

    normalized = (signal - signal.min()) / (signal.max() - signal.min() + 1e-6)

    for t in range(timesteps):
        spikes[t] = np.random.rand(len(signal)) < normalized * max_rate / timesteps

    return spikes.astype(float)
