import json
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd


class RetrievalDataset:
    def __init__(self, data: List[Dict], output_file: str = "retrieval_dataset.jsonl"):
        self.data = data
        self.output_file = output_file

    def generate_queries(self, patent: Dict) -> List[str]:
        queries = []
        
        if title := patent.get('title'):
            queries.append(title.strip())
        
        # 2. Add first sentence of abstract
        if abstract := patent.get('abstract'):
            first_sentence = abstract.split('.')[0] + '.'
            if first_sentence not in queries:
                queries.append(first_sentence)
        
        # 3. Add first 30 words of summary
        if summary := patent.get('summary'):
            summary_preview = ' '.join(summary.split()[:30])
            if summary_preview not in queries:
                queries.append(summary_preview)
        return queries

    def create_dataset(self) -> 'pd.DataFrame':
        
        testData = []
        for patent in self.data:
            doc_id = patent.get('publication_number', str(hash(str(patent))))
            queries = self.generate_queries(patent)
            
            for query in queries:
                testData.append({
                    "query": query,
                    "document_id": doc_id,
                    "title": patent.get('title', ''),
                    "abstract": patent.get('abstract', ''),
                    "summary": patent.get('summary', ''),
                    "relevance_score": 1.0
                })
        
        # Create DataFrame
        df = pd.DataFrame(testData)
        
        # Save to JSONL file
        output_file = Path(self.output_file).with_suffix('.jsonl')
        df.to_json(output_file, orient='records', lines=True)
        
        print(f"Saved {len(df)} query-document pairs to {output_file}")
        return df


if __name__ == "__main__":

    # Get patent data
    patent_data = get_IP_data(limit=1000)  # Adjust limit as needed
    
    # Create dataset
    dataset_creator = RetrievalDataset(patent_data)
    dataset = dataset_creator.create_dataset()