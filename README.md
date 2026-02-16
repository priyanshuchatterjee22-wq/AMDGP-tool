# AMDGP-tool
# 🧬 Drug-Gene Interaction Prioritizer

An AI/ML-based tool for predicting and prioritizing drug-gene interactions using ensemble machine learning.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![ML](https://img.shields.io/badge/ML-Ensemble-orange)]()

## 🎯 Key Features

- **Ensemble Learning**: Combines Random Forest, Gradient Boosting, and Neural Networks
- **Feature Engineering**: Automated extraction of drug and gene features
- **Confidence Scoring**: Provides interpretable confidence levels for predictions
- **Scalable**: Handles large-scale drug and gene databases
- **Visualizations**: Feature importance plots and ROC curves
- **Real-world Ready**: Demonstrated on cancer screening, personalized medicine, and drug repositioning

## 🚀 Quick Start

### Installation

```bash
# Clone or download this repository
git clone <your-repo-url>
cd drug-gene-prioritizer

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from drug_gene_prioritizer import DrugGeneInteractionPrioritizer
import pandas as pd

# 1. Initialize
prioritizer = DrugGeneInteractionPrioritizer()

# 2. Load your data
drug_data = pd.read_csv('your_drugs.csv')
gene_data = pd.read_csv('your_genes.csv')

# 3. Create features
drug_features = prioritizer.create_drug_features(drug_data)
gene_features = prioritizer.create_gene_features(gene_data)

# 4. Train (with your labeled interaction data)
X_train = # Your combined feature matrix
y_train = # Your labels (1=interact, 0=no interaction)
results = prioritizer.train(X_train, y_train)

# 5. Prioritize interactions
prioritized = prioritizer.prioritize_interactions(
    drug_data, 
    gene_data,
    top_k=100,
    threshold=0.6
)

# 6. Export results
prioritized.to_csv('prioritized_interactions.csv')
```

### Run Demo

```bash
# Run the complete demo with synthetic data
python drug_gene_prioritizer.py

# Run practical examples
python practical_examples.py
```

## 📊 Required Data Format

### Drug Data
Your drug CSV should contain these columns:
- `drug_id`: Unique identifier
- `drug_name`: Human-readable name
- `molecular_weight`: In Daltons
- `logP`: Lipophilicity (octanol-water partition coefficient)
- `h_bond_donors`: Number of hydrogen bond donors
- `h_bond_acceptors`: Number of hydrogen bond acceptors
- `rotatable_bonds`: Number of rotatable bonds
- `polar_surface_area`: Topological polar surface area
- `aromatic_rings`: Number of aromatic ring systems

**Example:**
```csv
drug_id,drug_name,molecular_weight,logP,h_bond_donors,h_bond_acceptors,rotatable_bonds,polar_surface_area,aromatic_rings
DRUG_001,Aspirin,180.2,1.19,1,4,3,63.6,1
DRUG_002,Ibuprofen,206.3,3.97,1,2,4,37.3,1
```

### Gene Data
Your gene CSV should contain these columns:
- `gene_id`: Unique identifier
- `gene_name`: Human-readable gene symbol
- `expression_level`: Expression level (e.g., FPKM, TPM)
- `protein_length`: Number of amino acids
- `binding_domains`: Number of ligand binding domains
- `phosphorylation_sites`: Number of phosphorylation sites
- `cellular_location_score`: Score 0-1 for cellular localization
- `pathway_connectivity`: Number of pathways gene participates in

**Example:**
```csv
gene_id,gene_name,expression_level,protein_length,binding_domains,phosphorylation_sites,cellular_location_score,pathway_connectivity
GENE_001,EGFR,12.5,1210,5,15,0.9,20
GENE_002,TP53,8.3,393,2,8,0.8,25
```

## 🔬 Use Cases

### 1. Cancer Drug Screening
Identify which cancer drugs are most likely to interact with specific oncogenes.

```python
# See practical_examples.py - example_cancer_drug_screening()
```

### 2. Personalized Medicine
Predict drug responses based on patient's genetic variations.

```python
# See practical_examples.py - example_personalized_medicine()
```

### 3. Drug Repositioning
Find new therapeutic uses for existing drugs.

```python
# See practical_examples.py - example_drug_repositioning()
```

## 📈 Performance Metrics

The tool provides multiple evaluation metrics:
- **Accuracy**: Overall prediction accuracy
- **AUC-ROC**: Area under ROC curve (ability to rank interactions)
- **Cross-validation score**: Model stability across different data splits
- **Feature importance**: Which features drive predictions

## 🧠 How It Works

### System Architecture

```
Drug Properties + Gene Properties
           ↓
    Feature Engineering
           ↓
  ┌──────────────────────┐
  │  Ensemble Models     │
  ├──────────────────────┤
  │ • Random Forest      │
  │ • Gradient Boosting  │
  │ • Neural Network     │
  └──────────────────────┘
           ↓
    Ensemble Voting
           ↓
  Prioritized Interactions
```

### Machine Learning Models

1. **Random Forest** (200 trees)
   - Provides feature importance
   - Robust to outliers
   - Handles non-linear relationships

2. **Gradient Boosting** (150 trees)
   - High predictive accuracy
   - Learns from previous errors
   - Good for imbalanced data

3. **Neural Network** (128-64-32 architecture)
   - Captures complex patterns
   - Non-linear feature combinations
   - Deep learning capabilities

### Feature Engineering

The tool automatically creates:
- **Individual features**: Drug and gene properties separately
- **Derived features**: Ratios and products of features
- **Interaction terms**: Cross-products of drug-gene features

## 📚 Documentation

- **DOCUMENTATION.md**: Comprehensive guide to logic, algorithms, and customization
- **Code comments**: Detailed explanations in source code
- **Examples**: See `practical_examples.py` for real-world scenarios

## 🎨 Customization

### Add Your Own Features

```python
def create_custom_drug_features(drug_data):
    # Add custom molecular descriptors
    drug_data['custom_feature'] = calculate_your_feature(drug_data)
    return drug_data
```

### Adjust Model Hyperparameters

```python
prioritizer = DrugGeneInteractionPrioritizer()
prioritizer.models['random_forest'].set_params(
    n_estimators=500,  # More trees
    max_depth=20       # Deeper trees
)
```

### Add New Models

```python
from sklearn.svm import SVC

prioritizer.models['svm'] = SVC(
    kernel='rbf',
    probability=True
)
```

## 🔧 Advanced Features

### Parallel Processing

```python
from joblib import Parallel, delayed

# Parallelize predictions
results = Parallel(n_jobs=-1)(
    delayed(predict_pair)(i, j) 
    for i in range(len(drugs)) 
    for j in range(len(genes))
)
```

### Model Persistence

```python
import pickle

# Save trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(prioritizer, f)

# Load model
with open('model.pkl', 'rb') as f:
    prioritizer = pickle.load(f)
```

### Deep Learning Extension

```python
# Add TensorFlow/PyTorch models
# See DOCUMENTATION.md for details
```

## 📊 Output Format

The prioritizer returns a DataFrame with:
- `drug_id`, `drug_name`: Drug identifiers
- `gene_id`, `gene_name`: Gene identifiers
- `interaction_score`: Probability of interaction (0-1)
- `confidence`: Confidence level (Very High, High, Medium, Low)
- `priority_rank`: Ranking by interaction score

## 🐛 Troubleshooting

### Low Accuracy
- Increase training data
- Add more informative features
- Adjust model hyperparameters

### Overfitting
- Reduce model complexity
- Add regularization
- Use more cross-validation

### Slow Predictions
- Use batch prediction
- Reduce number of features
- Implement caching

See **DOCUMENTATION.md** for detailed troubleshooting.

## 📖 References

1. **Lipinski's Rule of Five** - Drug-likeness criteria
2. **QSAR** - Quantitative Structure-Activity Relationship
3. **Ensemble Learning** - Combining multiple models
4. **PharmGKB** - Pharmacogenomics database
5. **DrugBank** - Drug and drug target database

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with scikit-learn, pandas, and matplotlib.

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check DOCUMENTATION.md for detailed explanations
- Review practical_examples.py for usage patterns

## 🎯 Future Enhancements

- [ ] Graph Neural Networks for molecular structures
- [ ] Integration with PubChem and DrugBank APIs
- [ ] Transfer learning from large public datasets
- [ ] Web interface for easy access
- [ ] Real-time prediction API
- [ ] Integration with electronic health records

---

**Happy Drug Discovery! 🧪🔬**
