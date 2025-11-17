
# ============================================================================
# 5. INFERENCE AND DECISION SUPPORT
# ============================================================================

class DisasterDecisionSupport:
    """Real-time decision support system for disaster response"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.preprocessor = DisasterDataPreprocessor()
    
    def analyze_situation(self, image: np.ndarray, report: str, 
                         sensors: Dict[str, float], lat: float, lon: float,
                         elevation: float, population_density: float) -> Dict:
        """
        Analyze disaster situation and provide recommendations
        """
        with torch.no_grad():
            # Preprocess inputs
            img_tensor = self.preprocessor.preprocess_satellite_image(image).unsqueeze(0)
            text_dict = self.preprocessor.preprocess_text_report(report)
            sensor_tensor = self.preprocessor.preprocess_sensor_data(sensors).unsqueeze(0)
            geo_tensor = self.preprocessor.preprocess_geodata(
                lat, lon, elevation, population_density
            ).unsqueeze(0)
            
            # Move to device
            batch = {
                'image': img_tensor.to(self.device),
                'text_ids': text_dict['input_ids'].unsqueeze(0).to(self.device),
                'text_mask': text_dict['attention_mask'].unsqueeze(0).to(self.device),
                'sensors': sensor_tensor.to(self.device),
                'geodata': geo_tensor.to(self.device)
            }
            
            # Get predictions
            outputs = self.model(batch)
            
            # Convert to probabilities
            severity_probs = F.softmax(outputs['severity'], dim=1)[0]
            evacuation_probs = F.softmax(outputs['evacuation'], dim=1)[0]
            priority_probs = F.softmax(outputs['priority'], dim=1)[0]
            
            # Get predictions
            severity_level = severity_probs.argmax().item()
            evacuation_needed = evacuation_probs.argmax().item() == 1
            priority_level = priority_probs.argmax().item()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                severity_level, evacuation_needed, priority_level, sensors
            )
            
            # Calculate confidence scores
            confidence = {
                'severity': severity_probs.max().item(),
                'evacuation': evacuation_probs.max().item(),
                'priority': priority_probs.max().item()
            }
            
            return {
                'severity': {
                    'level': severity_level,
                    'label': ['Minimal', 'Low', 'Moderate', 'High', 'Critical'][severity_level],
                    'confidence': confidence['severity']
                },
                'evacuation': {
                    'needed': evacuation_needed,
                    'confidence': confidence['evacuation']
                },
                'priority': {
                    'level': priority_level,
                    'label': ['Low', 'Medium', 'High'][priority_level],
                    'confidence': confidence['priority']
                },
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_recommendations(self, severity: int, evacuation: bool, 
                                 priority: int, sensors: Dict) -> List[str]:
        """Generate actionable recommendations based on predictions"""
        recommendations = []
        
        if severity >= 3:  # High or Critical
            recommendations.append("Deploy emergency response teams immediately")
            recommendations.append("Activate emergency operations center")
        
        if evacuation:
            recommendations.append("Initiate evacuation procedures for affected zones")
            recommendations.append("Establish evacuation routes and safe zones")
            recommendations.append("Coordinate with local transportation authorities")
        
        if priority >= 2:  # High priority
            recommendations.append("Request additional resources from neighboring regions")
            recommendations.append("Alert hospitals and medical facilities")
        
        # Sensor-specific recommendations
        if sensors.get('seismic_activity', 0) > 6.0:
            recommendations.append("Monitor aftershocks and structural integrity")
        
        if sensors.get('water_level', 0) > 10:
            recommendations.append("Deploy water rescue teams and equipment")
        
        if sensors.get('temperature', 0) > 300:
            recommendations.append("Deploy firefighting aircraft and ground crews")
        
        if sensors.get('wind_speed', 0) > 100:
            recommendations.append("Secure loose objects and reinforce structures")
        
        recommendations.append("Maintain continuous monitoring with satellite updates")
        recommendations.append("Establish communication with local authorities")
        
        return recommendations

