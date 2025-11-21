import torch
import torch.nn
from utils.net import Net
import onnx
from onnxsim import simplify
import onnxoptimizer

# model=Net('convnext_pico.d1_in1k',num_class=64,embeddingdim=512,mode='pred')
# model=Net('convnext_tiny.dinov3_lvd1689m',num_class=64,embeddingdim=512,mode='pred')
# model=Net('vit_base_patch16_dinov3.lvd_1689m',num_class=64,embeddingdim=512,mode='pred')
# model=Net('fastvit_mci3.apple_mclip2_dfndr2b',num_class=64,embeddingdim=512,mode='pred')
# model=Net('naflexvit_base_patch16_siglip.v2_webli',num_class=64,embeddingdim=512,mode='pred')
# model=Net('fasternet_t0.in1k',num_class=64,embeddingdim=512,mode='pred')
# model=Net('rdnet_small.nv_in1k',num_class=64,embeddingdim=512,mode='pred')
# model=Net('mambaout_femto.in1k',num_class=64,embeddingdim=512,mode='pred')
model=Net('tf_efficientnet_b1.ns_jft_in1k',num_class=64,embeddingdim=512,mode='pred')
model.load_state_dict(torch.load('./pt/tf_efficientnet_b1.pt',map_location='cpu'))
model.eval()

input_name = 'input'
output_name = 'output'
output_name1 = 'output1'


x = torch.randn(1,3,320,320,requires_grad=False)

torch.onnx.export(model, x, './onnx/best.onnx', input_names=[input_name], output_names=[output_name,output_name1], verbose=False,
                  dynamic_axes={
                        input_name:{0:'batch_size'},
                        output_name:{0:'batch_size'},
                        output_name1: {0: 'batch_size'}
                  },
                  opset_version=16,
                  do_constant_folding=True)



print('step 1 ok')
model = onnx.load('./onnx/best.onnx')
newmodel=onnxoptimizer.optimize(model)
model_simp, check = simplify(newmodel)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp,'./onnx/best-smi.onnx')
print('step 2 ok')


