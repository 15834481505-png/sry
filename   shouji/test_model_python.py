import onnxruntime as rt
import numpy as np

# 加载模型
session = rt.InferenceSession('model/model.onnx')

# 获取输入和输出信息
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f'输入名称: {input_name}')
print(f'输出名称: {output_name}')

# 创建测试输入
input_shape = session.get_inputs()[0].shape
print(f'输入形状: {input_shape}')

# 生成随机输入数据
input_data = np.random.rand(*input_shape).astype(np.float32)

# 运行推理
print('正在运行推理...')
outputs = session.run([output_name], {input_name: input_data})

# 分析输出
output = outputs[0]
print(f'输出形状: {output.shape}')
print(f'输出数据类型: {output.dtype}')
print(f'输出数据长度: {output.size}')

# 显示前20个数据点
print('前20个数据点:')
print(output.flatten()[:20])

# 分析输出格式
if len(output.shape) == 3:
    print('\n输出格式: [batch, channels, detections]')
    batch_size, num_channels, num_detections = output.shape
    print(f'批次大小: {batch_size}')
    print(f'通道数: {num_channels}')
    print(f'检测数: {num_detections}')
    print(f'类别数: {num_channels - 4}')
elif len(output.shape) == 2:
    print('\n输出格式: [detections, channels]')
    num_detections, num_channels = output.shape
    print(f'检测数: {num_detections}')
    print(f'通道数: {num_channels}')
    print(f'类别数: {num_channels - 4}')
else:
    print('\n未知的输出格式')
    print(f'输出形状: {output.shape}')

print('测试完成')