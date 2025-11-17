import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# ============================================================================
# 7. DEMONSTRATION
# ============================================================================

def create_synthetic_data(num_samples: int = 100) -> List[Dict]:
    """Create synthetic disaster data for demonstration"""
    disaster_types = ['earthquake', 'flood', 'wildfire', 'hurricane', 'tornado']
    data = []
    
    for i in range(num_samples):
        disaster_type = np.random.choice(disaster_types)
        
        # Generate synthetic data based on disaster type
        if disaster_type == 'earthquake':
            severity = np.random.randint(2, 5)
            sensors = {
                'seismic_activity': np.random.uniform(5.0, 9.0),
                'temperature': np.random.uniform(15, 25),
                'wind_speed': np.random.uniform(5, 20)
            }
            report = f"Earthquake magnitude {sensors['seismic_activity']:.1f} detected"
        elif disaster_type == 'flood':
            severity = np.random.randint(1, 5)
            sensors = {
                'water_level': np.random.uniform(5, 30),
                'humidity': np.random.uniform(70, 100),
                'temperature': np.random.uniform(10, 30)
            }
            report = f"Flooding with water level at {sensors['water_level']:.1f}m"
        elif disaster_type == 'wildfire':
            severity = np.random.randint(2, 5)
            sensors = {
                'temperature': np.random.uniform(200, 500),
                'wind_speed': np.random.uniform(20, 80),
                'humidity': np.random.uniform(10, 30)
            }
            report = f"Wildfire spreading rapidly with high temperatures"
        else:
            severity = np.random.randint(1, 4)
            sensors = {
                'wind_speed': np.random.uniform(50, 150),
                'pressure': np.random.uniform(920, 1000),
                'temperature': np.random.uniform(15, 30)
            }
            report = f"Severe weather event with high winds"
        
        data.append({
            'image': np.random.rand(224, 224, 3).astype(np.float32) * 255,
            'report': report,
            'sensors': sensors,
            'lat': np.random.uniform(-60, 60),
            'lon': np.random.uniform(-180, 180),
            'elevation': np.random.uniform(0, 2000),
            'population_density': np.random.uniform(10, 5000),
            'severity': severity,
            'evacuation_needed': 1 if severity >= 3 else 0,
            'response_priority': min(severity - 1, 2)
        })
    
    return data


def main_demo():
    """Main demonstration function"""
    print("Initializing Multimodal Disaster Response System...")
    print("=" * 70)
    
    # Initialize components
    preprocessor = DisasterDataPreprocessor()
    model = DisasterResponseModel(feature_dim=512)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    print("\nGenerating synthetic disaster data...")
    data = create_synthetic_data(num_samples=200)
    dataset = DisasterMultimodalDataset(data, preprocessor)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Training
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    trainer = DisasterResponseTrainer(model, device)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Severity Acc: {val_metrics['severity_acc']:.2%}")
        print(f"Val Evacuation Acc: {val_metrics['evacuation_acc']:.2%}")
        print(f"Val Priority Acc: {val_metrics['priority_acc']:.2%}")
    
    # Inference demonstration
    print("\n" + "=" * 70)
    print("INFERENCE DEMONSTRATION")
    print("=" * 70)
    
    decision_support = DisasterDecisionSupport(model, device)
    report_gen = DisasterReportGenerator(decision_support)
    
    # Test case: Severe earthquake
    test_image = np.random.rand(224, 224, 3).astype(np.float32) * 255
    test_report = "Major earthquake detected, magnitude 7.8, widespread damage"
    test_sensors = {
        'seismic_activity': 7.8,
        'temperature': 22.0,
        'wind_speed': 15.0,
        'humidity': 60.0,
        'pressure': 1013.0
    }
    
    print("\nAnalyzing disaster situation...")
    result = decision_support.analyze_situation(
        image=test_image,
        report=test_report,
        sensors=test_sensors,
        lat=34.05,
        lon=-118.24,
        elevation=100,
        population_density=3000
    )
    
    print("\n" + report_gen.generate_report(result, "Los Angeles County, CA"))
    
    print("\n" + "=" * 70)
    print("System ready for real-time disaster response!")
    print("=" * 70)


if __name__ == "__main__":
    # Note: This requires actual installation of dependencies:
    # pip install torch torchvision transformers numpy
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║   MULTIMODAL AI FOR DISASTER RESPONSE - RESEARCH IMPLEMENTATION   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    This implementation demonstrates a complete multimodal AI system for
    disaster response that integrates:
    
    • Satellite Imagery (CNNs with ResNet-50)
    • Text Reports (Transformers with BERT)
    • Sensor Data (Multi-Layer Perceptrons)
    • Geospatial Information (Coordinate encoding)
    
    The system uses cross-attention fusion to combine information from
    all modalities and provides real-time decision support including:
    - Severity assessment
    - Evacuation recommendations
    - Response prioritization
    
    Key Features:
    ✓ Multi-task learning architecture
    ✓ Attention-based multimodal fusion
    ✓ Real-time inference capabilities
    ✓ Comprehensive decision support
    ✓ Automated report generation
    """)
    
    # Uncomment to run demonstration
    # main_demo()