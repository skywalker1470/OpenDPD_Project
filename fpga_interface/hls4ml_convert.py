import hls4ml
import onnx

model_onnx = onnx.load('dpd_lstm.onnx')

config = hls4ml.utils.config_from_onnx_model(
    model_onnx,
    granularity='name',
    default_precision='ap_fixed<16,6>',   # 16-bit fixed point, 6 integer bits
)

hls_model = hls4ml.converters.convert_from_onnx_model(
    model_onnx,
    hls_config=config,
    output_dir='hls_project',
    part='xc7a35tcpg236-1',   # Arty A7-35T — low-cost dev board
    clock_period=10,           # 100 MHz
)

hls_model.compile()           # requires Vivado HLS installed
report = hls_model.build(csim=True)
print(report)