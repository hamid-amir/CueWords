# CueWords: Gender Agreement Analysis in Language Models

A research project that analyzes gender agreement patterns in fine-tuned language models using context mixing and value patching interpretability techniques.

## Overview

This project investigates how language models learn and maintain gender agreement patterns through:
- **Data Creation**: Generating gender agreement datasets from Wikipedia biographies
- **Fine-tuning**: Training various transformer models (BERT, RoBERTa, GPT-2) on gender agreement tasks
- **Context Mixing Analysis**: Measuring how well models maintain gender consistency across different contexts
- **Value Patching**: Analyzing and modifying model representations to understand gender encoding

## Project Structure

```
CueWords/
├── data_creation/          # Dataset generation scripts
├── fine_tuning/            # Model training and fine-tuning
├── context_mixing/         # Context mixing analysis tools
├── value_patching/         # Value patching implementation
├── data/                   # Generated datasets
├── results_vp/             # Value patching results
├── results_cm/             # Context mixing results
├── notebooks/              # Jupyter notebooks for analysis
└── figs/                   # Generated figures and visualizations
```

## Key Features

- **Multi-Model Support**: Works with BERT, and GPT-2 architectures
- **Gender Agreement Dataset**: Automatically generated from WikiBio dataset
- **Context Mixing Toolkit**: Integrated analysis tools for measuring model consistency
- **Value Patching**: Advanced techniques for analyzing and modifying model representations
- **Comprehensive Evaluation**: Multiple metrics and analysis approaches

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hamid-amir/CueWords
cd CueWords
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download SpaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Quick Start

Run the complete pipeline:
```bash
./runAll.sh
```

Or run individual components:

1. **Generate dataset**:
```bash
python3 data_creation/gender_agreement.py
```

2. **Fine-tune models**:
```bash
python3 fine_tuning/train.py
```

3. **Calculate context mixing scores**:
```bash
python3 context_mixing/main.py
```
<!-- 
## Important Dependencies

- `datasets==2.19.1` - Hugging Face datasets library
- `spacy==3.7.4` - Natural language processing
- `transformers==4.38.2` - Hugging Face transformers library
- `torch` - PyTorch (for model training)
- `numpy` - Numerical computing -->

## Research Applications

This project is designed for researchers studying:
- Gender bias in language models
- Model interpretability and analysis
- Context mixing and consistency in NLP
- Value patching and representation analysis
- Fine-tuning effects on model behavior


If you use this code in your research, please cite the following paper and acknowledge this repository:

```bibtex
@inproceedings{amirzadeh-etal-2024-language,
  title     = "How Language Models Prioritize Contextual Grammatical Cues?",
  author    = "Amirzadeh, Hamidreza and
               Alishahi, Afra and
               Mohebbi, Hosein",
  editor    = "Belinkov, Yonatan and
               Kim, Najoung and
               Jumelet, Jaap and
               Mohebbi, Hosein and
               Mueller, Aaron and
               Chen, Hanjie",
  booktitle = "Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP",
  month     = nov,
  year      = "2024",
  address   = "Miami, Florida, US",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2024.blackboxnlp-1.21/",
  doi       = "10.18653/v1/2024.blackboxnlp-1.21",
  pages     = "315--336",
}
```

## License

MIT


## Contact

For questions or issues, please open an issue on the repository.
