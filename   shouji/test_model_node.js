// 测试模型输出格式
const ort = require('onnxruntime-node');

async function testModel() {
    try {
        // 加载模型
        console.log('正在加载模型...');
        const model = await ort.InferenceSession.create('model/model.onnx');
        
        console.log('模型加载成功');
        console.log('输入名称:', model.inputNames);
        console.log('输出名称:', model.outputNames);
        
        // 创建测试输入
        const inputShape = [1, 3, 640, 640];
        const inputSize = inputShape.reduce((a, b) => a * b);
        const inputData = new Float32Array(inputSize);
        
        for (let i = 0; i < inputSize; i++) {
            inputData[i] = Math.random();
        }
        
        const inputTensor = new ort.Tensor('float32', inputData, inputShape);
        
        // 运行推理
        console.log('正在运行推理...');
        const results = await model.run({ images: inputTensor });
        
        console.log('推理完成');
        console.log('输出结果:', results);
        
        // 分析输出
        for (const [name, tensor] of Object.entries(results)) {
            console.log(`\n输出 ${name}:`);
            console.log('形状:', tensor.dims);
            console.log('数据类型:', tensor.type);
            console.log('数据长度:', tensor.data.length);
            console.log('前10个数据:', Array.from(tensor.data).slice(0, 10));
        }
        
    } catch (error) {
        console.error('测试失败:', error);
    }
}

testModel();