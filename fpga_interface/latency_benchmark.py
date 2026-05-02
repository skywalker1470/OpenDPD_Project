import time, torch, numpy as np
from backbones.my_dpd_backbone import MyLSTMDPD

model = MyLSTMDPD(input_size=2, hidden_size=32)
model.load_state_dict(torch.load('results_my_dpd/my_lstm_dpd_best.pth'))
model.eval()

x = torch.randn(1, 15, 2)
runs = 1000

# CPU latency
times = []
with torch.no_grad():
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - t0)

print(f"CPU inference latency: {np.mean(times)*1e6:.1f} us")
# FPGA latency comes from hls4ml synthesis report