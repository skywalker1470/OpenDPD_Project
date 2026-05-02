import torch
from backbones.my_dpd_backbone import MyLSTMDPD

model = MyLSTMDPD(input_size=2, hidden_size=32)
model.load_state_dict(torch.load('results_my_dpd/my_lstm_dpd_best.pth', map_location='cpu'))
model.eval()

dummy = torch.randn(1, 15, 2)   # batch=1, seq_len=15, IQ=2
torch.onnx.export(
    model, dummy, 'dpd_lstm.onnx',
    input_names=['iq_input'],
    output_names=['dpd_output'],
    dynamic_axes={'iq_input': {0: 'batch'}},
    opset_version=14,
)
print("Exported dpd_lstm.onnx")