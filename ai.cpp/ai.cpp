#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>

// 定义矩阵类用于矩阵运算
class Matrix {
public:
    std::vector<std::vector<double>> data;
    int rows, cols;

    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(r, std::vector<double>(c, 0.0));
    }

    Matrix(const std::vector<std::vector<double>>& d) : data(d) {
        rows = data.size();
        cols = (rows > 0) ? data[0].size() : 0;
    }

    // 矩阵乘法
    Matrix dot(const Matrix& other) const {
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    // 矩阵转置
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    // 矩阵加法
    Matrix add(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    // 元素乘法 (Hadamard product)
    Matrix elementwiseMultiply(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }
    
    // 标量乘法
    Matrix multiply(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }
    
    // 应用函数到每个元素
    Matrix apply(std::function<double(double)> func) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }
    
    // 矩阵行归一化 (Softmax按行)
    Matrix softmax() const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            double maxVal = *std::max_element(data[i].begin(), data[i].end());
            double sum = 0.0;
            
            // 计算指数和
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = exp(data[i][j] - maxVal);
                sum += result.data[i][j];
            }
            
            // 归一化
            for (int j = 0; j < cols; j++) {
                result.data[i][j] /= sum;
            }
        }
        return result;
    }
    
    // 层归一化
    Matrix layerNorm(double epsilon = 1e-6) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            // 计算平均值
            double mean = 0.0;
            for (int j = 0; j < cols; j++) {
                mean += data[i][j];
            }
            mean /= cols;
            
            // 计算方差
            double var = 0.0;
            for (int j = 0; j < cols; j++) {
                var += pow(data[i][j] - mean, 2);
            }
            var /= cols;
            
            // 归一化
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = (data[i][j] - mean) / sqrt(var + epsilon);
            }
        }
        return result;
    }
};

// Transformer架构
class Transformer {
private:
    int d_model;      // 模型维度
    int nhead;        // 注意力头数
    int dim_feedforward; // 前馈网络维度
    int seq_len;      // 序列长度
    double dropout_rate; // dropout比率
    
    // 随机数生成器
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    
    // 参数
    struct TransformerParams {
        // 多头注意力参数
        std::vector<Matrix> query_weights;  // 每个头的Q矩阵
        std::vector<Matrix> key_weights;    // 每个头的K矩阵
        std::vector<Matrix> value_weights;  // 每个头的V矩阵
        Matrix attn_output_weight;          // 多头注意力输出投影
        
        // 前馈网络参数
        Matrix fc1_weight;
        Matrix fc2_weight;
        
        // Layer Norm参数 (简化，仅使用简单的层归一化)
        
        // 偏置项
        std::vector<Matrix> query_bias;
        std::vector<Matrix> key_bias;
        std::vector<Matrix> value_bias;
        Matrix attn_output_bias;
        Matrix fc1_bias;
        Matrix fc2_bias;
        
        // 添加默认构造函数
        TransformerParams() 
            : attn_output_weight(1, 1), fc1_weight(1, 1), fc2_weight(1, 1),
              attn_output_bias(1, 1), fc1_bias(1, 1), fc2_bias(1, 1) {}
    };
    
    TransformerParams params;
    
    // 位置编码
    Matrix positionalEncoding;
    
public:
    Transformer(int model_dim, int heads, int ff_dim, int sequence_length, double dropout = 0.1) 
        : d_model(model_dim), nhead(heads), dim_feedforward(ff_dim), seq_len(sequence_length),
          dropout_rate(dropout), gen(std::random_device()()), params(), positionalEncoding(1, 1) {
        
        dis = std::uniform_real_distribution<double>(-0.1, 0.1);
        
        // 初始化参数
        initializeParameters();
        
        // 预计算位置编码
        computePositionalEncoding();
    }
    
    void initializeParameters() {
        // 初始化多头注意力的权重
        for (int i = 0; i < nhead; i++) {
            Matrix q_weight(d_model, d_model / nhead);
            Matrix k_weight(d_model, d_model / nhead);
            Matrix v_weight(d_model, d_model / nhead);
            
            // 随机初始化
            for (int r = 0; r < q_weight.rows; r++) {
                for (int c = 0; c < q_weight.cols; c++) {
                    q_weight.data[r][c] = dis(gen);
                    k_weight.data[r][c] = dis(gen);
                    v_weight.data[r][c] = dis(gen);
                }
            }
            
            params.query_weights.push_back(q_weight);
            params.key_weights.push_back(k_weight);
            params.value_weights.push_back(v_weight);
            
            // 偏置
            Matrix q_bias(1, d_model / nhead);
            Matrix k_bias(1, d_model / nhead);
            Matrix v_bias(1, d_model / nhead);
            
            for (int c = 0; c < q_bias.cols; c++) {
                q_bias.data[0][c] = dis(gen);
                k_bias.data[0][c] = dis(gen);
                v_bias.data[0][c] = dis(gen);
            }
            
            params.query_bias.push_back(q_bias);
            params.key_bias.push_back(k_bias);
            params.value_bias.push_back(v_bias);
        }
        
        // 多头注意力输出权重
        params.attn_output_weight = Matrix(d_model, d_model);
        for (int r = 0; r < params.attn_output_weight.rows; r++) {
            for (int c = 0; c < params.attn_output_weight.cols; c++) {
                params.attn_output_weight.data[r][c] = dis(gen);
            }
        }
        
        // 多头注意力输出偏置
        params.attn_output_bias = Matrix(1, d_model);
        for (int c = 0; c < params.attn_output_bias.cols; c++) {
            params.attn_output_bias.data[0][c] = dis(gen);
        }
        
        // 前馈网络权重
        params.fc1_weight = Matrix(d_model, dim_feedforward);
        params.fc2_weight = Matrix(dim_feedforward, d_model);
        
        for (int r = 0; r < params.fc1_weight.rows; r++) {
            for (int c = 0; c < params.fc1_weight.cols; c++) {
                params.fc1_weight.data[r][c] = dis(gen);
            }
        }
        
        for (int r = 0; r < params.fc2_weight.rows; r++) {
            for (int c = 0; c < params.fc2_weight.cols; c++) {
                params.fc2_weight.data[r][c] = dis(gen);
            }
        }
        
        // 前馈网络偏置
        params.fc1_bias = Matrix(1, dim_feedforward);
        params.fc2_bias = Matrix(1, d_model);
        
        for (int c = 0; c < params.fc1_bias.cols; c++) {
            params.fc1_bias.data[0][c] = dis(gen);
        }
        
        for (int c = 0; c < params.fc2_bias.cols; c++) {
            params.fc2_bias.data[0][c] = dis(gen);
        }
    }
    
    // 计算位置编码
    void computePositionalEncoding() {
        positionalEncoding = Matrix(seq_len, d_model);
        
        for (int pos = 0; pos < seq_len; pos++) {
            for (int i = 0; i < d_model; i += 2) {
                positionalEncoding.data[pos][i] = sin(pos / pow(10000, (2 * i) / d_model));
                if (i + 1 < d_model) {
                    positionalEncoding.data[pos][i + 1] = cos(pos / pow(10000, (2 * i) / d_model));
                }
            }
        }
    }
    
    // 应用ReLU激活函数
    double relu(double x) {
        return x > 0 ? x : 0;
    }
    
    // 多头自注意力机制
    Matrix multiheadAttention(const Matrix& input) {
        std::vector<Matrix> head_outputs;
        
        for (int h = 0; h < nhead; h++) {
            // 计算Q, K, V
            Matrix Q = input.dot(params.query_weights[h]);
            Matrix K = input.dot(params.key_weights[h]);
            Matrix V = input.dot(params.value_weights[h]);
            
            // 添加偏置
            for (int i = 0; i < Q.rows; i++) {
                for (int j = 0; j < Q.cols; j++) {
                    Q.data[i][j] += params.query_bias[h].data[0][j];
                    K.data[i][j] += params.key_bias[h].data[0][j];
                    V.data[i][j] += params.value_bias[h].data[0][j];
                }
            }
            
            // 计算注意力分数 (Q * K^T / sqrt(d_k))
            Matrix scores = Q.dot(K.transpose());
            double scale = sqrt(d_model / nhead);
            scores = scores.multiply(1.0 / scale);
            
            // 应用Softmax得到注意力权重
            Matrix attention_weights = scores.softmax();
            
            // 应用注意力权重到值矩阵
            Matrix head_output = attention_weights.dot(V);
            head_outputs.push_back(head_output);
        }
        
        // 拼接所有头的输出
        Matrix concat_heads(input.rows, d_model);
        int head_dim = d_model / nhead;
        
        for (int i = 0; i < input.rows; i++) {
            for (int h = 0; h < nhead; h++) {
                for (int j = 0; j < head_dim; j++) {
                    concat_heads.data[i][h * head_dim + j] = head_outputs[h].data[i][j];
                }
            }
        }
        
        // 投影到输出空间
        Matrix output = concat_heads.dot(params.attn_output_weight);
        
        // 添加输出偏置
        for (int i = 0; i < output.rows; i++) {
            for (int j = 0; j < output.cols; j++) {
                output.data[i][j] += params.attn_output_bias.data[0][j];
            }
        }
        
        return output;
    }
    
    // 前馈网络
    Matrix feedForward(const Matrix& input) {
        // 第一层 + ReLU
        Matrix hidden = input.dot(params.fc1_weight);
        
        // 添加偏置
        for (int i = 0; i < hidden.rows; i++) {
            for (int j = 0; j < hidden.cols; j++) {
                hidden.data[i][j] += params.fc1_bias.data[0][j];
            }
        }
        
        // 应用ReLU
        hidden = hidden.apply([this](double x) { return relu(x); });
        
        // 第二层
        Matrix output = hidden.dot(params.fc2_weight);
        
        // 添加偏置
        for (int i = 0; i < output.rows; i++) {
            for (int j = 0; j < output.cols; j++) {
                output.data[i][j] += params.fc2_bias.data[0][j];
            }
        }
        
        return output;
    }
    
    // 前向传播
    Matrix forward(const Matrix& input) {
        // 添加位置编码
        Matrix x = input;
        for (int i = 0; i < std::min(x.rows, positionalEncoding.rows); i++) {
            for (int j = 0; j < x.cols; j++) {
                x.data[i][j] += positionalEncoding.data[i][j];
            }
        }
        
        // 自注意力 + 残差连接 + 层归一化
        Matrix attn_output = multiheadAttention(x);
        Matrix residual1 = x.add(attn_output);
        Matrix norm1 = residual1.layerNorm();
        
        // 前馈网络 + 残差连接 + 层归一化
        Matrix ff_output = feedForward(norm1);
        Matrix residual2 = norm1.add(ff_output);
        Matrix output = residual2.layerNorm();
        
        return output;
    }
    
    // 简单的预测函数
    Matrix predict(const Matrix& input) {
        return forward(input);
    }
};

// 展示Transformer输出
void printMatrix(const Matrix& mat, const std::string& name) {
    std::cout << name << " (" << mat.rows << "x" << mat.cols << "):" << std::endl;
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            std::cout << mat.data[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // 定义模型参数
    int d_model = 16;         // 模型维度
    int nhead = 2;            // 注意力头数
    int dim_feedforward = 32; // 前馈网络维度
    int seq_len = 5;          // 最大序列长度
    
    // 创建Transformer模型
    Transformer transformer(d_model, nhead, dim_feedforward, seq_len);
    
    std::cout << "Transformer模型初始化完成" << std::endl;
    std::cout << "模型维度: " << d_model << std::endl;
    std::cout << "注意力头数: " << nhead << std::endl;
    std::cout << "前馈网络维度: " << dim_feedforward << std::endl;
    std::cout << "最大序列长度: " << seq_len << std::endl << std::endl;
    
    // 创建输入序列（随机数据）
    Matrix input(3, d_model); // 3个token的序列，每个token是d_model维
    
    // 随机初始化输入
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            input.data[i][j] = dis(gen);
        }
    }
    
    // 输出输入序列
    printMatrix(input, "输入序列");
    
    // 通过Transformer处理序列
    Matrix output = transformer.predict(input);
    
    // 输出结果
    printMatrix(output, "Transformer输出");
    
    std::cout << "Transformer演示完成！" << std::endl;
    
    return 0;
}
