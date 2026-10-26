# Summer 2026 Research Implementations

From-scratch PyTorch implementations of landmark deep learning papers, written during my Summer 2026 machine learning research at FIU. Each notebook maps its code back to the relevant sections and equations of the original paper.

| Paper | Notebook | Task | Result |
|---|---|---|---|
| Krizhevsky et al., 2012, *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet) | [AlexNet.ipynb](AlexNet.ipynb) | CIFAR-10 image classification | 80.2% val accuracy after 15 epochs (~3M params) |
| He et al., 2015, *Deep Residual Learning for Image Recognition* (ResNet) | [ResNet.ipynb](ResNet.ipynb) | CIFAR-10 image classification | 79.6% val accuracy after 10 epochs (ResNet-20, ~272K params) |
| Vaswani et al., 2017, *Attention Is All You Need* (Transformer) | [Transformer.ipynb](Transformer.ipynb) | Sequence reversal (seq2seq) | 100% token accuracy with greedy decoding |
| Greff et al., 2015, *LSTM: A Search Space Odyssey* | [LSTM.ipynb](LSTM.ipynb) · [write-up (PDF)](LSTM.pdf) | Long-range copy task | Vanilla, CIFG and NP variants reach near-identical loss |

## Projects

### AlexNet
A scaled-down AlexNet for 32×32 CIFAR-10 images: five conv layers with ReLU, max-pooling and dropout, trained with SGD and momentum, data augmentation (random crop and horizontal flip, Section 4.1), and learning-rate decay when validation loss plateaus. Includes training curves, per-class accuracy, top-5 predictions on test images and a confusion matrix.

### ResNet
ResNet-20 built from residual `BasicBlock`s (Section 3.2) with batch normalization after every convolution and projection shortcuts when dimensions change. Trained with the paper's SGD settings (momentum 0.9, weight decay 1e-4, LR 0.1 divided by 10 on plateau). After only 10 epochs it matches the Mini-AlexNet's accuracy with about a tenth of the parameters (272K vs. 3M), which shows what skip connections buy.

### Transformer
The full encoder-decoder Transformer built from its parts: scaled dot-product attention, multi-head attention, position-wise feed-forward layers, sinusoidal positional encoding, causal masking, and the paper's warmup learning-rate schedule (Section 5.3). Trained with teacher forcing to reverse sequences, then evaluated with true autoregressive greedy decoding.

### LSTM: A Search Space Odyssey
Hand-written LSTM cells (no `nn.LSTM`), so every gate equation from the paper is visible in code:
- **Vanilla**: forget gate plus peephole connections
- **CIFG**: input and forget gates coupled (f = 1 − i)
- **NP**: no peepholes

All three are trained with the same seed and hyperparameters on a copy task that requires holding tokens in memory across a delay. Their final losses (0.5351 / 0.5344 / 0.5369) match the paper's finding that these simplifications don't significantly hurt performance. The notebook also plots the forget gate's activations over the delay period. The [PDF write-up](LSTM.pdf) summarizes the paper and the results.

## Paper review: concept-based interpretability
[interpretability_review.pptx](interpretability_review.pptx) is a 16-slide review of three concept-based approaches to interpretable deep learning, with my take on each: Concept Bottleneck Models (Koh et al., ICML 2020), Concept Activation Vectors (TCAV, Kim et al. 2018; PCBM, Yuksekgonul et al. 2023), and Concept Embedding Models (Espinosa Zarlenga et al., NeurIPS 2022).

## Running the notebooks
Every notebook runs top to bottom in Google Colab or locally. They use a CUDA GPU when one is available and download CIFAR-10 automatically.

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```
