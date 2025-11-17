
# ============================================================================
# 2. DATASET CLASS
# ============================================================================

class DisasterMultimodalDataset(Dataset):
    """Dataset for multimodal disaster response data"""
    
    def __init__(self, data_list: List[Dict], preprocessor: DisasterDataPreprocessor):
        self.data = data_list
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Process each modality
        image = self.preprocessor.preprocess_satellite_image(sample['image'])
        text = self.preprocessor.preprocess_text_report(sample['report'])
        sensors = self.preprocessor.preprocess_sensor_data(sample['sensors'])
        geodata = self.preprocessor.preprocess_geodata(
            sample['lat'], sample['lon'], 
            sample['elevation'], sample['population_density']
        )
        
        # Labels: severity (0-4), evacuation_needed (0/1), response_priority (0-2)
        labels = torch.tensor([
            sample['severity'],
            sample['evacuation_needed'],
            sample['response_priority']
        ], dtype=torch.long)
        
        return {
            'image': image,
            'text_ids': text['input_ids'],
            'text_mask': text['attention_mask'],
            'sensors': sensors,
            'geodata': geodata,
            'labels': labels
        }
