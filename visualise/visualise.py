
# ============================================================================
# 6. VISUALIZATION AND REPORTING
# ============================================================================

class DisasterReportGenerator:
    """Generate comprehensive disaster assessment reports"""
    
    def __init__(self, decision_support: DisasterDecisionSupport):
        self.ds = decision_support
    
    def generate_report(self, analysis_result: Dict, location_name: str) -> str:
        """Generate a formatted text report"""
        report = []
        report.append("=" * 70)
        report.append("DISASTER RESPONSE ASSESSMENT REPORT")
        report.append("=" * 70)
        report.append(f"\nLocation: {location_name}")
        report.append(f"Timestamp: {analysis_result['timestamp']}")
        report.append(f"\n{'SEVERITY ASSESSMENT':-^70}")
        report.append(f"Level: {analysis_result['severity']['label']}")
        report.append(f"Confidence: {analysis_result['severity']['confidence']:.2%}")
        
        report.append(f"\n{'EVACUATION STATUS':-^70}")
        report.append(f"Evacuation Needed: {'YES' if analysis_result['evacuation']['needed'] else 'NO'}")
        report.append(f"Confidence: {analysis_result['evacuation']['confidence']:.2%}")
        
        report.append(f"\n{'RESPONSE PRIORITY':-^70}")
        report.append(f"Priority: {analysis_result['priority']['label']}")
        report.append(f"Confidence: {analysis_result['priority']['confidence']:.2%}")
        
        report.append(f"\n{'RECOMMENDED ACTIONS':-^70}")
        for i, rec in enumerate(analysis_result['recommendations'], 1):
            report.append(f"{i}. {rec}")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)
