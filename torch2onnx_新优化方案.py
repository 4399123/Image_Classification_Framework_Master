import torch
import torch.nn
import onnx
import os
import sys
from onnxslim import slim
from polygraphy.backend.onnx import fold_constants
from utils.net import Net


def main():
    # ----------------------------------------------------------------
    # 2. 模型初始化 (保持你的原始逻辑)
    # ----------------------------------------------------------------
    # model=Net('convnext_pico.d1_in1k',num_class=64,embeddingdim=512,mode='pred')
    # model=Net('convnext_tiny.dinov3_lvd1689m',num_class=64,embeddingdim=512,mode='pred')
    # model=Net('vit_base_patch16_dinov3.lvd_1689m',num_class=64,embeddingdim=512,mode='pred')
    # model=Net('fastvit_mci3.apple_mclip2_dfndr2b',num_class=64,embeddingdim=512,mode='pred')
    # model=Net('naflexvit_base_patch16_siglip.v2_webli',num_class=64,embeddingdim=512,mode='pred')
    # model=Net('fasternet_t0.in1k',num_class=64,embeddingdim=512,mode='pred')
    # model=Net('rdnet_small.nv_in1k',num_class=64,embeddingdim=512,mode='pred')
    # model=Net('mambaout_femto.in1k',num_class=64,embeddingdim=512,mode='pred')
    model=Net('tf_efficientnet_b1.ns_jft_in1k',num_class=64,embeddingdim=512,mode='pred')
    model.load_state_dict(torch.load('./pt/tf_efficientnet_b1.pt', map_location='cpu'))
    model.eval()

    # ----------------------------------------------------------------
    # 3. 导出设置
    # ----------------------------------------------------------------
    input_name = 'input'
    output_name = 'output'
    output_name1 = 'output1'

    # 临时文件路径 (原始导出)
    raw_onnx_path = './onnx/best_raw.onnx'
    # 最终文件路径 (优化后)
    final_onnx_path = './onnx/best-smi.onnx'

    os.makedirs(os.path.dirname(final_onnx_path), exist_ok=True)

    # x = torch.randn(1, 3, 320, 320, requires_grad=False)
    x = torch.randint(0, 256, (1, 3, 320, 320), dtype=torch.uint8)
    print(f"Step 1: Exporting raw ONNX to {raw_onnx_path}...")
    torch.onnx.export(
        model,
        x,
        raw_onnx_path,
        input_names=[input_name],
        output_names=[output_name, output_name1],
        verbose=False,
        dynamic_axes={
            input_name: {0: 'batch_size'},
            output_name: {0: 'batch_size'},
            output_name1: {0: 'batch_size'}
        },
        opset_version=16,
        do_constant_folding=True
    )

    slimmed_model = slim(raw_onnx_path, model_check=True)
    print(' -> onnxslim optimization complete.')


    # polygraphy 负责最后的一致性检查和数值折叠
    final_model = fold_constants(slimmed_model)

    # 保存最终模型
    onnx.save(final_model, final_onnx_path)
    print(f' -> Optimization successful! Saved to: {final_onnx_path}')

    # 打印文件大小变化
    raw_size = os.path.getsize(raw_onnx_path) / 1024 / 1024
    final_size = os.path.getsize(final_onnx_path) / 1024 / 1024
    print(f" -> Size reduction: {raw_size:.2f} MB -> {final_size:.2f} MB")

    # 清理中间文件
    if os.path.exists(raw_onnx_path):
        os.remove(raw_onnx_path)


if __name__ == '__main__':
    main()