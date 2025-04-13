# C++实现的Transformer模型

本项目实现了一个简化版的Transformer架构，使用C++语言编写。Transformer是自然语言处理领域的一种强大模型，在机器翻译、文本生成和语言理解等任务中表现出色。

## 模型架构

该实现包含Transformer的核心组件：

1. **多头自注意力机制 (Multi-head Self-Attention)**：
   - 允许模型关注输入序列中的不同位置
   - 支持多个注意力头并行计算，每个头关注不同的表示子空间
   - 实现了缩放点积注意力(Scaled Dot-Product Attention)

2. **位置编码 (Positional Encoding)**：
   - 使用正弦和余弦函数生成位置编码
   - 为模型提供序列中位置的信息

3. **前馈神经网络 (Feed-Forward Network)**：
   - 两层全连接网络
   - 使用ReLU激活函数

4. **层归一化 (Layer Normalization)**：
   - 加速训练
   - 稳定深层网络中的激活值

5. **残差连接 (Residual Connection)**：
   - 帮助梯度流动
   - 缓解训练深层网络的困难

## 关键类

### Matrix

自定义矩阵类，提供矩阵运算支持：

- 矩阵乘法
- 矩阵转置
- 元素级乘法
- 矩阵加法
- Softmax操作
- 层归一化

### Transformer

核心Transformer实现：

- 构造函数接受模型维度、注意力头数、前馈网络维度和序列长度
- 提供前向传播、多头注意力和前馈网络等功能

## 使用方法

```cpp
// 初始化Transformer
int d_model = 16;         // 模型维度
int nhead = 2;            // 注意力头数
int dim_feedforward = 32; // 前馈网络维度
int seq_len = 5;          // 最大序列长度

Transformer transformer(d_model, nhead, dim_feedforward, seq_len);

// 创建输入
Matrix input(3, d_model); // 3个token的序列，每个token是d_model维
// ... 填充输入数据 ...

// 前向传播
Matrix output = transformer.predict(input);
```

## 简化设计

相比完整的Transformer实现，此版本做了以下简化：

1. 没有实现编码器-解码器架构，仅实现了编码器部分
2. 未包含掩码机制
3. 没有实现训练功能
4. 未加入Dropout层
5. 层归一化使用了简化版本

## 扩展方向

可以从以下方面扩展该实现：

1. 添加训练功能，包括反向传播和优化器
2. 实现解码器部分
3. 添加词嵌入层
4. 加入批处理支持
5. 增加保存和加载模型的功能

## 示例代码

主程序演示了创建Transformer模型、处理输入序列并生成输出的过程：

```cpp
int main() {
    // 定义模型参数
    int d_model = 16;
    int nhead = 2;
    int dim_feedforward = 32;
    int seq_len = 5;
    
    // 创建Transformer模型
    Transformer transformer(d_model, nhead, dim_feedforward, seq_len);
    
    // 创建输入序列
    Matrix input(3, d_model);
    
    // 初始化输入
    // ... 初始化代码 ...
    
    // 通过Transformer处理序列
    Matrix output = transformer.predict(input);
    
    // 输出结果
    // ... 输出代码 ...
    
    return 0;
}
```
