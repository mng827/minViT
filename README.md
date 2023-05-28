A minimal implementation of Vision Transformers for educational purposes, inspired
by Andrei Karpathy's minGPT.

Requirements:
- PyTorch
- Hydra

Files:
- `model.py` contains the definition for the ViT model.
- `main.py` contains a simple training loop to train on the CIFAR10 dataset.

To train the Vision Transformer from scratch with the CIFAR10 dataset, run:
`python main.py`. This uses the config file in `config/config.yaml`
and achieves an accuracy of 77% on the test set. Fine-tuning a pretrained
model should give better performances.

Feedback welcome!
