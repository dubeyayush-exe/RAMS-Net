# ============================================================================
# 1. DATA PREPROCESSING MODULES
# ============================================================================

class DisasterDataPreprocessor:
    """Preprocessing pipeline for multimodal disaster data"""
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def preprocess_satellite_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess satellite imagery
        Args:
            image: RGB satellite image (H, W, 3)
        Returns:
            Preprocessed tensor (3, 224, 224)
        """
        # Normalize and resize
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear')
        image = (image - image.mean()) / (image.std() + 1e-5)
        return image.squeeze(0)
    
    def preprocess_text_report(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Preprocess text reports using BERT tokenizer
        Args:
            text: Emergency report text
            max_length: Maximum sequence length
        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def preprocess_sensor_data(self, sensors: Dict[str, float]) -> torch.Tensor:
        """
        Preprocess sensor readings (seismic, weather, water level, etc.)
        Args:
            sensors: Dictionary of sensor readings
        Returns:
            Normalized sensor tensor
        """
        # Define expected sensor types and normalization ranges
        sensor_ranges = {
            'temperature': (-50, 500),  # Celsius
            'humidity': (0, 100),  # Percentage
            'wind_speed': (0, 200),  # km/h
            'seismic_activity': (0, 10),  # Richter scale
            'water_level': (-10, 50),  # meters
            'air_quality': (0, 500),  # AQI
            'pressure': (900, 1100)  # hPa
        }
        
        normalized = []
        for key, (min_val, max_val) in sensor_ranges.items():
            val = sensors.get(key, 0.0)
            norm_val = (val - min_val) / (max_val - min_val)
            normalized.append(norm_val)
        
        return torch.tensor(normalized, dtype=torch.float32)
    
    def preprocess_geodata(self, lat: float, lon: float, elevation: float,
                          population_density: float) -> torch.Tensor:
        """
        Preprocess geospatial data
        Args:
            lat: Latitude
            lon: Longitude
            elevation: Elevation in meters
            population_density: People per km²
        Returns:
            Normalized geospatial tensor
        """
        # Normalize coordinates and other geo features
        lat_norm = (lat + 90) / 180
        lon_norm = (lon + 180) / 360
        elev_norm = (elevation + 500) / 9500  # Assume range -500 to 9000m
        pop_norm = np.log1p(population_density) / 15  # Log scale
        
        return torch.tensor([lat_norm, lon_norm, elev_norm, pop_norm], dtype=torch.float32)

