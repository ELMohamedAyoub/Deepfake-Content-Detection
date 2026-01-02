# Advanced Multimodal Disinformation Detection with Transformers

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.35+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-ready deep learning system using state-of-the-art transformers (CLIP + BERT) for detecting fake news through combined image and text analysis.

## Project Versions

This repository contains two implementations showcasing evolution from baseline to advanced architecture:

### Version 1: Baseline (EfficientNet + GRU)
**File**: `multimodal_disinformation_detection.ipynb`
- **Image**: EfficientNetB0 (ImageNet pre-trained)
- **Text**: Bidirectional GRU + word embeddings
- **Fusion**: Late fusion with concatenation
- **Performance**: 94.2% accuracy, 0.982 ROC-AUC

### Version 2: Advanced (CLIP + BERT + Cross-Attention) ⭐
**File**: `multimodal_disinformation_detection_v2.ipynb`
- **Image**: CLIP Vision Transformer (400M image-text pairs)
- **Text**: BERT-base (contextual embeddings)
- **Fusion**: Cross-attention mechanism
- **Dataset**: Fakeddit-ready (real multimodal fake news)
- **Training**: Mixed precision, advanced LR scheduling

## Why Version 2 is Superior

| Component | V1 (Baseline) | V2 (Transformers) | Why Better |
|-----------|---------------|-------------------|------------|
| **Image Model** | EfficientNetB0 | CLIP ViT-B/32 | Vision-language pre-training on 400M pairs |
| **Text Model** | Bi-GRU | BERT-base | Contextual vs. static embeddings |
| **Fusion** | Concatenation | Cross-Attention | Learns which text relates to which image regions |
| **Dataset** | Synthetic (LFW+CIFAR) | Fakeddit (real) | Real-world fake news patterns |
| **Training** | Standard | Mixed precision | 2x faster, same accuracy |
| **Embeddings** | 128 + 64 = 192D | 512 + 768 = 1280D | Richer representations |

### Key Achievements

**Version 1 (Baseline):**
- 94.2% accuracy on synthetic dataset
- Production API with FastAPI
- Grad-CAM interpretability
- MLflow experiment tracking

**Version 2 (Advanced):**
- State-of-the-art transformer architecture
- Real dataset support (Fakeddit)
- Cross-attention for multimodal reasoning
- Advanced training techniques (AdamW, warmup, mixed precision)
- Attention visualization

## Architecture

**Image Branch:**
- EfficientNetB0 (pretrained on ImageNet, frozen weights)
- Global Average Pooling
- Dense layers with regularization

**Text Branch:**
- Word Embeddings (10k vocabulary)
- Bidirectional GRU
- Global Max Pooling

**Fusion Module:**
- Concatenate features
- Dense layers with L2 regularization and dropout
- Binary classification output

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/multimodal-disinformation-detection.git
cd multimodal-disinformation-detection

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

**Option 1: Google Colab (Recommended)**
1. Upload `multimodal_disinformation_detection.ipynb` to Google Colab
2. Runtime → Change runtime type → GPU
3. Run all cells

**Option 2: Local Jupyter**
```bash
jupyter notebook multimodal_disinformation_detection.ipynb
```

## Project Structure

```
multimodal-disinformation-detection/
├── multimodal_disinformation_detection.ipynb  # Main notebook
├── ELMOUSATI_exam_TP_PHASE_1.ipynb           # Research Phase 1
├── ELMOUSATI_exam_TP_PHASE_2.ipynb           # Research Phase 2
├── ELMOUSATI_exam_TP_PHASE_3.ipynb           # Research Phase 3
├── requirements.txt                           # Dependencies
├── README.md                                  # This file
├── .gitignore                                # Git ignore rules
└── LICENSE                                    # MIT License
```

## Technical Details

### Dataset Strategy
Due to the lack of public multimodal disinformation datasets, this project uses a hybrid approach:
- **Real Content (Class 0)**: LFW faces + authentic news headlines
- **Fake Content (Class 1)**: CIFAR-10 objects + fake news headlines

This creates a proxy dataset that simulates semantic consistency patterns found in real disinformation.

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 94.2% |
| Precision | 91.7% |
| Recall | 94.1% |
| F1-Score | 92.9% |
| ROC-AUC | 0.982 |
| PR-AUC | 0.975 |

### Key Features

**MLOps Integration:**
- MLflow experiment tracking
- Model versioning and registry
- Hyperparameter logging
- Artifact management

**Production Deployment:**
- FastAPI REST API
- Prometheus metrics
- Health check endpoints
- Auto-generated documentation

**Model Interpretability:**
- Grad-CAM visual explanations
- Attention heatmaps
- Confusion matrix analysis

**Monitoring:**
- Data drift detection (Evidently)
- Performance tracking
- Statistical tests

## Usage Examples

### Training
The notebook includes complete training pipeline with:
- Data loading and preprocessing
- Model architecture definition
- Training with callbacks
- Comprehensive evaluation

### API Deployment
After training, deploy the model as a REST API:
```python
# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

Example prediction request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "text=Breaking news headline" \
  -F "image=@image.jpg"
```

Response:
```json
{
  "risk_score": 0.87,
  "predicted_label": "Fake",
  "confidence": 0.74,
  "timestamp": "2026-01-02T10:30:00"
}
```

## Methodology

### Late Fusion Architecture
This project employs late fusion (feature-level fusion) rather than early or decision-level fusion because:
1. Allows specialized processing for each modality
2. Better captures modality-specific patterns
3. More flexible than early fusion
4. More efficient than decision-level fusion

### Training Strategy
- Transfer learning with frozen EfficientNetB0
- L2 regularization (0.001)
- Dropout (0.3-0.5) for regularization
- Data augmentation (flip, rotation, zoom)
- Early stopping based on validation loss
- Learning rate scheduling

### Evaluation Approach
Prioritized **recall over precision** because in content moderation:
- False negatives (missing fake content) are costly
- False positives (flagging real content) can be reviewed

## Limitations

1. **Synthetic Dataset**: Uses proxy sources rather than real disinformation
2. **English Only**: Text processing limited to English language
3. **Binary Classification**: No nuanced scoring of misleading levels
4. **Static Images**: Does not process video or animated content
5. **Computational Cost**: Requires GPU for efficient inference

## Future Work

- [ ] Train on real multimodal disinformation datasets (Fakeddit, etc.)
- [ ] Implement multilingual support
- [ ] Add temporal analysis for video content
- [ ] Deploy transformer-based encoders (CLIP, BERT)
- [ ] Implement active learning pipeline
- [ ] Add text-level interpretability (SHAP)
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] A/B testing framework

## Research Context

This project evolved through three research phases:

**Phase 1**: Baseline implementation with MobileNetV2 + Bi-LSTM (90% accuracy)

**Phase 2**: Upgraded to EfficientNetB0 + Bi-GRU with synthetic correlation strategy (95% accuracy)

**Phase 3**: Production system with advanced regularization, interpretability, and deployment (94% accuracy with better generalization)

## Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: EfficientNet, OpenCV
- **NLP**: Word embeddings, Bidirectional GRU
- **MLOps**: MLflow, Evidently
- **API**: FastAPI, Uvicorn
- **Monitoring**: Prometheus
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Interpretability**: Grad-CAM, SHAP

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Mohamed Ayoub ELMOUSATI**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## Acknowledgments

- EfficientNet architecture by Google Research
- CIFAR-10 dataset by Alex Krizhevsky
- LFW dataset by University of Massachusetts, Amherst
- Fake News dataset contributors
- Open source community

## Citations

If you use this work in your research, please cite:

```bibtex
@software{elmousati2026multimodal,
  author = {Elmousati, Mohamed Ayoub},
  title = {Multimodal Disinformation Detection System},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/multimodal-disinformation-detection}
}
```

## References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
2. Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder.
3. Selvaraju, R. R., et al. (2017). Grad-cam: Visual explanations from deep networks.
4. Nakamura, K., et al. (2019). Fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection.

---

**Star this repository if you find it useful!**
