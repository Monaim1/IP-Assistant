from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Dict, Tuple

class CPCMatcher:
    def __init__(self):
        # Initialize the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('AI-Growth-Lab/PatentSBERTa')
        self.model = AutoModel.from_pretrained('AI-Growth-Lab/PatentSBERTa')
        self.model.eval()  # Set the model to evaluation mode
        
        # CPC sections with descriptions
        self.cpc_sections = {
            'A': 'Human Necessities - Agriculture, food, personal, health',
            'B': 'Performing Operations - Processing, transporting',
            'C': 'Chemistry - Metallurgy, materials, chemical engineering',
            'D': 'Textiles - Paper, fibers, fabrics',
            'E': 'Fixed Constructions - Building, civil engineering',
            'F': 'Mechanical Engineering - Engines, lighting, heating',
            'G': 'Physics - Computing, measuring, optics',
            'H': 'Electricity - Electrical engineering, circuits',
            'Y': 'General - Cross-sectional technologies'
        }
        
        # Pre-compute embeddings for CPC section descriptions
        self.cpc_embeddings = self._get_cpc_embeddings()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        inputs = self.tokenizer(text, return_tensors='pt', 
                              padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token as the sentence embedding
        return outputs.last_hidden_state[:, 0, :].numpy()[0]
    def _get_cpc_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all CPC section descriptions."""
        return {section: self._get_embedding(desc) 
                for section, desc in self.cpc_sections.items()}
    
    def find_matching_cpc(self, query: str) -> str:
        query_embedding = self._get_embedding(query)
        return max(
            self.cpc_embeddings.items(),
            key=lambda x: np.dot(query_embedding, x[1]) / 
                        (np.linalg.norm(query_embedding) * np.linalg.norm(x[1]))
        )[0]

# Example usage
if __name__ == "__main__":
    matcher = CPCMatcher()
    
    test_queries = [
        "neural network for image recognition",
        "new pharmaceutical compound",
        "solar panel mounting system",
        "database query optimization"
    ]
    
    for query in test_queries:
        print("\n" + "="*80)
        print(matcher.find_matching_cpc(query))
        print("="*80)