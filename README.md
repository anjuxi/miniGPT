# Training a Small GPT Language Model with minGPT

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About

This project is built on [minGPT](https://github.com/karpathy/minGPT) by [Andrej Karpathy](https://github.com/karpathy). It trains a character-level language model from scratch, walking through data prep, model definition, training, and inference.

The training corpus is *Journey to the West* (`xyj.txt`). The resulting model generates text in the style of classical Chinese literature.

## Project Structure

```
.
├── minGPT.ipynb    # main notebook: model definition, training, inference
└── xyj.txt         # training data: full text of Journey to the West
```

## Features

- Character-level autoregressive language model
- Transformer architecture with multi-head self-attention
- Full training loop: data preprocessing, training, loss monitoring
- Text generation from a trained model

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- NumPy >= 1.24
- Jupyter Notebook

## Quick Start

1. Open the notebook:

   ```bash
   jupyter notebook minGPT.ipynb
   ```

2. Run the cells in order.

## Model Architecture

The notebook implements:

- **CausalSelfAttention** — masked causal self-attention
- **MLP** — feed-forward network
- **Block** — Transformer block (attention + MLP + residual + LayerNorm)
- **GPT** — the full model

### Hyperparameters

```python
@dataclass # default model config
class Config:
    batch_size: int = 64;
    block_size: int = 128;
    n_embd: int = 512;
    vocab_size: int = None;
    n_layer: int = 2;
    n_head: int = 2;
    dropout: float = 0.1;
    bias: bool = True;
    grad_norm_clip: float = 1.0; # gradient clipping threshold
config = Config();
```

## Training Data

`xyj.txt` contains roughly 740,000 Chinese characters. The model learns the syntax and phrasing of classical Chinese.

## Usage

### Training

Run the training cells. They will:

1. Load and encode the training data
2. Initialize the GPT model
3. Run the training loop
4. Print the loss periodically

### Generating Text

After training:

```python
# Load a saved checkpoint if needed
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
print(f"generated: {generated}");
```

## License

This project is based on [minGPT](https://github.com/karpathy/minGPT) and follows the MIT license:

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

## Author

**Dr. Ailan Anjuxi (谙弆悕)**

- Email: [anjuxi.ME@outlook.com](mailto:anjuxi.ME@outlook.com)
- SIP: [sip:anjuxi@sip.linphone.org](sip:anjuxi@sip.linphone.org)

## References

- [minGPT - Andrej Karpathy](https://github.com/karpathy/minGPT)
