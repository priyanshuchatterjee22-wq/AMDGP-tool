"""
Drug-Gene Interaction Prioritizer
==================================
An AI/ML-based tool for predicting and prioritizing drug-gene interactions
using ensemble machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DrugGeneInteractionPrioritizer:
    """
    Main class for drug-gene interaction prediction and prioritization.
    
    Uses ensemble learning combining:
    - Random Forest for feature importance
    - Gradient Boosting for high accuracy
    - Neural Network for complex pattern recognition
    """
    
    def __init__(self, random_state: int = 42):
        
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Initialize ensemble models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=7,
                random_state=random_state
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=random_state,
                early_stopping=True
            )
        }
        
        self.is_trained = False
        self.feature_names = None
        self.feature_importance = None
        
    def create_drug_features(self, drug_data: pd.DataFrame) -> np.ndarray:
        """
        Create feature vectors for drugs.
        
        Expected columns in drug_data:
        - molecular_weight: float
        - logP: float (lipophilicity)
        - h_bond_donors: int
        - h_bond_acceptors: int
        - rotatable_bonds: int
        - polar_surface_area: float
        - aromatic_rings: int
        
        Args:
            drug_data: DataFrame with drug properties
            
        Returns:
            Feature matrix for drugs
        """
        features = []
        
        for _, drug in drug_data.iterrows():
            drug_features = [
                drug.get('molecular_weight', 0),
                drug.get('logP', 0),
                drug.get('h_bond_donors', 0),
                drug.get('h_bond_acceptors', 0),
                drug.get('rotatable_bonds', 0),
                drug.get('polar_surface_area', 0),
                drug.get('aromatic_rings', 0),
                # Derived features
                drug.get('h_bond_donors', 0) + drug.get('h_bond_acceptors', 0),  # Total H-bonds
                drug.get('molecular_weight', 0) / max(drug.get('rotatable_bonds', 1), 1)  # MW/flexibility
            ]
            features.append(drug_features)
        
        return np.array(features)
    
    def create_gene_features(self, gene_data: pd.DataFrame) -> np.ndarray:
        """
        Create feature vectors for genes.
        
        Expected columns in gene_data:
        - expression_level: float
        - protein_length: int
        - binding_domains: int
        - phosphorylation_sites: int
        - cellular_location_score: float
        - pathway_connectivity: int
        
        Args:
            gene_data: DataFrame with gene properties
            
        Returns:
            Feature matrix for genes
        """
        features = []
        
        for _, gene in gene_data.iterrows():
            gene_features = [
                gene.get('expression_level', 0),
                gene.get('protein_length', 0),
                gene.get('binding_domains', 0),
                gene.get('phosphorylation_sites', 0),
                gene.get('cellular_location_score', 0),
                gene.get('pathway_connectivity', 0),
                # Derived features
                gene.get('binding_domains', 0) / max(gene.get('protein_length', 1), 1),  # Domain density
                gene.get('expression_level', 0) * gene.get('pathway_connectivity', 0)  # Functional impact
            ]
            features.append(gene_features)
        
        return np.array(features)
    
    def combine_features(self, drug_features: np.ndarray, gene_features: np.ndarray) -> np.ndarray:
        """
        Combine drug and gene features for interaction prediction.
        
        Creates interaction features through:
        - Concatenation
        - Element-wise products (interaction terms)
        - Statistical aggregations
        
        Args:
            drug_features: Drug feature matrix
            gene_features: Gene feature matrix
            
        Returns:
            Combined feature matrix for drug-gene pairs
        """
        # Ensure same number of samples
        n_samples = min(len(drug_features), len(gene_features))
        drug_features = drug_features[:n_samples]
        gene_features = gene_features[:n_samples]
        
        # Concatenate features
        combined = np.hstack([drug_features, gene_features])
        
        # Add interaction terms (element-wise products of key features)
        # Interaction between drug molecular weight and gene expression
        interaction_1 = (drug_features[:, 0:1] * gene_features[:, 0:1])
        
        # Interaction between drug lipophilicity and gene binding domains
        interaction_2 = (drug_features[:, 1:2] * gene_features[:, 2:3])
        
        # Combine all features
        all_features = np.hstack([combined, interaction_1, interaction_2])
        
        return all_features
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict:
        """
        Train the ensemble models.
        
        Args:
            X: Feature matrix (drug-gene pairs)
            y: Labels (1 = interaction, 0 = no interaction)
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training metrics
        """
        print("Training Drug-Gene Interaction Prioritizer...")
        print(f"Dataset size: {X.shape[0]} interactions, {X.shape[1]} features")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Store feature names
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Train each model
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            val_score = model.score(X_val_scaled, y_val)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # ROC-AUC
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            
            results[name] = {
                'train_accuracy': train_score,
                'val_accuracy': val_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'auc': auc
            }
            
            print(f"  Train Acc: {train_score:.4f}")
            print(f"  Val Acc: {val_score:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  AUC: {auc:.4f}")
        
        # Calculate feature importance from Random Forest
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.models['random_forest'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        print("\nTraining completed!")
        
        return results
    
    def predict_interaction(self, X: np.ndarray, return_proba: bool = True) -> np.ndarray:
        """
        Predict drug-gene interactions using ensemble voting.
        
        Args:
            X: Feature matrix for drug-gene pairs
            return_proba: Return probabilities instead of binary predictions
            
        Returns:
            Predictions or probability scores
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = []
        for model in self.models.values():
            if return_proba:
                pred = model.predict_proba(X_scaled)[:, 1]
            else:
                pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Ensemble: average probabilities or majority vote
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def prioritize_interactions(
        self,
        drug_data: pd.DataFrame,
        gene_data: pd.DataFrame,
        top_k: Optional[int] = None,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Prioritize drug-gene interactions based on predicted scores.
        
        Args:
            drug_data: DataFrame with drug information and features
            gene_data: DataFrame with gene information and features
            top_k: Return top K interactions (if None, return all above threshold)
            threshold: Minimum probability threshold
            
        Returns:
            DataFrame with prioritized interactions
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")
        
        print("Generating predictions for all drug-gene pairs...")
        
        # Create feature matrices
        drug_features = self.create_drug_features(drug_data)
        gene_features = self.create_gene_features(gene_data)
        
        # Generate all drug-gene pairs
        results = []
        for i, drug_row in drug_data.iterrows():
            for j, gene_row in gene_data.iterrows():
                # Combine features for this pair
                drug_feat = drug_features[i].reshape(1, -1)
                gene_feat = gene_features[j].reshape(1, -1)
                combined = self.combine_features(drug_feat, gene_feat)
                
                # Predict interaction probability
                prob = self.predict_interaction(combined, return_proba=True)[0]
                
                if prob >= threshold:
                    results.append({
                        'drug_id': drug_row.get('drug_id', f'drug_{i}'),
                        'drug_name': drug_row.get('drug_name', f'Drug_{i}'),
                        'gene_id': gene_row.get('gene_id', f'gene_{j}'),
                        'gene_name': gene_row.get('gene_name', f'Gene_{j}'),
                        'interaction_score': prob,
                        'confidence': self._calculate_confidence(prob),
                        'priority_rank': 0  # Will be set after sorting
                    })
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print("No interactions found above threshold.")
            return results_df
        
        results_df = results_df.sort_values('interaction_score', ascending=False)
        results_df['priority_rank'] = range(1, len(results_df) + 1)
        
        # Return top K if specified
        if top_k is not None:
            results_df = results_df.head(top_k)
        
        print(f"Found {len(results_df)} prioritized interactions")
        
        return results_df.reset_index(drop=True)
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calculate confidence level based on probability."""
        if probability >= 0.9:
            return "Very High"
        elif probability >= 0.75:
            return "High"
        elif probability >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get top important features for interaction prediction.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet.")
        
        return self.feature_importance.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 15, figsize: Tuple = (10, 6)):
        """Plot feature importance."""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet.")
        
        top_features = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top Feature Importances for Drug-Gene Interactions')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature importance plot saved to 'feature_importance.png'")
    
    def plot_roc_curves(self, X_test: np.ndarray, y_test: np.ndarray, figsize: Tuple = (10, 6)):
        """Plot ROC curves for all models."""
        X_test_scaled = self.scaler.transform(X_test)
        
        plt.figure(figsize=figsize)
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Drug-Gene Interaction Models')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("ROC curves saved to 'roc_curves.png'")


def generate_synthetic_data(n_drugs: int = 100, n_genes: int = 50, interaction_rate: float = 0.3):
    """
    Generate synthetic drug and gene data for demonstration.
    
    Args:
        n_drugs: Number of drugs to generate
        n_genes: Number of genes to generate
        interaction_rate: Proportion of drug-gene pairs that interact
        
    Returns:
        Tuple of (drug_data, gene_data, interactions)
    """
    np.random.seed(42)
    
    # Generate drug data
    drug_data = pd.DataFrame({
        'drug_id': [f'DRUG_{i:04d}' for i in range(n_drugs)],
        'drug_name': [f'DrugName_{i}' for i in range(n_drugs)],
        'molecular_weight': np.random.normal(350, 100, n_drugs).clip(100, 800),
        'logP': np.random.normal(2.5, 1.5, n_drugs).clip(-2, 6),
        'h_bond_donors': np.random.poisson(2, n_drugs),
        'h_bond_acceptors': np.random.poisson(4, n_drugs),
        'rotatable_bonds': np.random.poisson(5, n_drugs),
        'polar_surface_area': np.random.normal(80, 30, n_drugs).clip(0, 200),
        'aromatic_rings': np.random.poisson(2, n_drugs)
    })
    
    # Generate gene data
    gene_data = pd.DataFrame({
        'gene_id': [f'GENE_{i:04d}' for i in range(n_genes)],
        'gene_name': [f'GeneName_{i}' for i in range(n_genes)],
        'expression_level': np.random.lognormal(2, 1, n_genes),
        'protein_length': np.random.normal(400, 150, n_genes).clip(50, 1500).astype(int),
        'binding_domains': np.random.poisson(3, n_genes),
        'phosphorylation_sites': np.random.poisson(5, n_genes),
        'cellular_location_score': np.random.uniform(0, 1, n_genes),
        'pathway_connectivity': np.random.poisson(8, n_genes)
    })
    
    # Generate interaction labels
    # Create realistic patterns: drugs with certain properties interact with genes with certain properties
    interactions = []
    for i in range(n_drugs):
        for j in range(n_genes):
            # Create interaction logic based on features
            drug_score = (drug_data.loc[i, 'logP'] > 2) + (drug_data.loc[i, 'h_bond_acceptors'] > 3)
            gene_score = (gene_data.loc[j, 'binding_domains'] > 2) + (gene_data.loc[j, 'expression_level'] > 5)
            
            # Probability based on feature compatibility
            base_prob = interaction_rate
            if drug_score >= 1 and gene_score >= 1:
                prob = base_prob * 2
            else:
                prob = base_prob * 0.5
            
            interaction = 1 if np.random.random() < prob else 0
            interactions.append(interaction)
    
    return drug_data, gene_data, np.array(interactions)


if __name__ == "__main__":
    print("=" * 80)
    print("Drug-Gene Interaction Prioritizer - Demo")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    drug_data, gene_data, _ = generate_synthetic_data(n_drugs=100, n_genes=50)
    print(f"   Generated {len(drug_data)} drugs and {len(gene_data)} genes")
    
    # Initialize prioritizer
    print("\n2. Initializing prioritizer...")
    prioritizer = DrugGeneInteractionPrioritizer()
    
    # Create features and labels for training
    print("\n3. Creating feature matrices...")
    drug_features = prioritizer.create_drug_features(drug_data)
    gene_features = prioritizer.create_gene_features(gene_data)
    
    # Create training data (all drug-gene pairs)
    X_train = []
    y_train = []
    
    for i in range(len(drug_data)):
        for j in range(len(gene_data)):
            drug_feat = drug_features[i].reshape(1, -1)
            gene_feat = gene_features[j].reshape(1, -1)
            combined = prioritizer.combine_features(drug_feat, gene_feat)
            X_train.append(combined[0])
            
            # Simulate realistic interaction labels
            drug_score = (drug_data.loc[i, 'logP'] > 2) + (drug_data.loc[i, 'h_bond_acceptors'] > 3)
            gene_score = (gene_data.loc[j, 'binding_domains'] > 2) + (gene_data.loc[j, 'expression_level'] > 5)
            
            if drug_score >= 1 and gene_score >= 1:
                prob = 0.6
            else:
                prob = 0.2
            
            y_train.append(1 if np.random.random() < prob else 0)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"   Created {len(X_train)} training examples")
    print(f"   Positive interactions: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    
    # Train models
    print("\n4. Training models...")
    results = prioritizer.train(X_train, y_train)
    
    # Show feature importance
    print("\n5. Top 10 Important Features:")
    print(prioritizer.get_feature_importance(10))
    
    # Prioritize interactions
    print("\n6. Prioritizing drug-gene interactions...")
    prioritized = prioritizer.prioritize_interactions(
        drug_data.head(20),  # Test with subset
        gene_data.head(10),
        top_k=15,
        threshold=0.5
    )
    
    print("\n7. Top Prioritized Interactions:")
    print(prioritized.to_string())
    
    # Generate plots
    print("\n8. Generating visualizations...")
    prioritizer.plot_feature_importance()
    
    # Save results
    prioritized.to_csv('prioritized_interactions.csv', index=False)
    print("\n✓ Results saved to 'prioritized_interactions.csv'")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

