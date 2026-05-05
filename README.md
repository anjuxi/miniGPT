# 使用 minGPT 训练自己的小型 GPT 语言模型

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目简介

本项目基于 [Andrej Karpathy](https://github.com/karpathy) 的 [minGPT](https://github.com/karpathy/minGPT)，旨在从零开始训练一个属于自己的字符级语言模型。通过本项目，您可以深入了解数据处理、模型搭建、训练和测试的完整流程。

项目使用《西游记》（xyj.txt）作为训练语料，训练一个能够生成中文古典文学风格文本的 GPT 模型。

## 项目结构

```
.
├── minGPT.ipynb    # 主程序：包含完整的模型定义、训练和推理代码
└── xyj.txt         # 训练数据：《西游记》全文
```

## 功能特性

- **字符级语言模型**：基于字符级别的自回归语言模型
- **Transformer 架构**：实现了多头自注意力机制
- **完整的训练流程**：包含数据预处理、模型训练、损失监控
- **文本生成**：支持使用训练好的模型生成文本

## 环境依赖

- Python >= 3.11
- PyTorch >= 2.0
- NumPy >= 1.24
- Jupyter Notebook

## 快速开始

1. 打开 Jupyter Notebook：

   ```bash
   jupyter notebook minGPT.ipynb
   ```

2. 按顺序运行 notebook 中的代码单元格即可

## 模型架构

本项目实现了以下核心组件：

- **CausalSelfAttention**: 因果自注意力机制（带掩码）
- **MLP**: 多层感知机（前馈网络）
- **Block**: Transformer 块（注意力 + MLP + 残差连接 + LayerNorm）
- **GPT**: 完整的 GPT 模型

### 超参数配置

```python
@dataclass # 默认模型配置。
class Config:
    batch_size: int = 64;
    block_size: int = 128;
    n_embd: int = 512;
    vocab_size: int = None;
    n_layer: int = 2;
    n_head: int = 2;
    dropout: float = 0.1;
    bias: bool = True;
    grad_norm_clip: float = 1.0; # 梯度裁剪阈值。
config = Config();
```

## 训练数据

本项目使用《西游记》作为训练语料（xyj.txt），包含约 74 万个中文字符。模型将学习中文古典文学的语法结构和表达方式。

## 使用方法

### 训练模型

运行 notebook 中的训练代码块，模型将自动：

1. 加载并编码训练数据
2. 初始化 GPT 模型
3. 进行迭代训练
4. 定期输出损失值

### 文本生成

训练完成后，使用以下代码生成文本：

```python
# 加载保存的模型（如果需要）。
# model.load_state_dict(torch.load(os.path.join(WORK_DIR, "model.pt")));
model.eval();

def generate_text(prompt, max_tokens=5, temperature=0.8, top_k=10):
    context = torch.tensor(
        [train_dataset.stoi.get(s, 0) for s in prompt],
        dtype=torch.long
    )[None, ...].to(device);

    with torch.no_grad():
        y = model.generate(context, max_tokens, temperature=temperature, do_sample=True, top_k=top_k)[0];

    completion = ''.join([train_dataset.itos[int(i)] for i in y]);
    return completion;


prompt = "孙";

generated = generate_text(prompt, max_tokens=100, temperature=0.8);
print(f"生成结果: {generated}");
```

## 许可证

本项目基于 [minGPT](https://github.com/karpathy/minGPT) 开发，遵循 MIT 许可证：

```
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 作者信息

**谙弆悕博士（Ailan Anjuxi）**

- 邮箱：[anjuxi.ME@outlook.com](mailto:anjuxi.ME@outlook.com)
- SIP 电话：[sip:anjuxi@sip.linphone.org](sip:anjuxi@sip.linphone.org)

## 参考资料

- [minGPT - Andrej Karpathy](https://github.com/karpathy/minGPT)

  
