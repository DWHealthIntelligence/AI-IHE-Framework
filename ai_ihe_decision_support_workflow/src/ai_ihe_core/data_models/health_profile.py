"""
Health Status Descirption: HealthQuestionnaire, DiseaseRisk, HealthBehavior

@author: DanWu and Agreewithu (Ruixin Dai)
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from .questionnaire_info import HealthQuestionnaire

@dataclass
class DiseaseRiskProfile:
    """
    Disease Early Prediction
    Based on Mutil-variable Time Series Deep Learning Model
    """
    # disease risks (e.g. {'Type2Diabetes': 0.85, 'Hypertension': 0.42})
    risk_scores: Dict[str, float] = field(default_factory=dict)
    
    # corresponding risk level (e.g. {'Type2Diabetes': 'High Risk', 'Hypertension': 'Low Risk'})
    risk_levels: Dict[str, str] = field(default_factory=dict)

@dataclass
class BehaviorPatternProfile:
    """
    Health Behavior
    Based on Time Series Culsting and Stochastic Process Analysis
    """
    # engagement (e.g. 'Highly Active', 'Declining Engagement')
    engagement_cluster: str 
    
    # multi-dimensional behavior trajectory preference distribution based on Markov model
    # (e.g. {'Task_Completion': 0.4, 'Health_Report_Viewing': 0.3, 'Knowledge_Learning': 0.2, 'System_Interaction': 0.1})
    behavior_trail_distribution: Dict[str, float] = field(default_factory=dict)

    # behavior preference setting threshold
    behavior_preference_threshold: float
    
    # core behavioral transition probability matrix
    transition_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class HealthStatusProfile:
    """
    Health Status Profile
    Defined by HEIRO
    """
    participant_id: str
    
    # 1. structured health information (questionnaire)
    structured_info: HealthQuestionnaire
    
    # 2. predicted disease risk (derived from historical physical examination records)
    disease_risks: DiseaseRiskProfile
    
    # 3. health behavior patterns (derived from mHealth software records)
    behavior_patterns: BehaviorPatternProfile
    
    def to_heiro_features(self) -> Dict[str, Any]:
        """
        Convert into HEIRO (Health Examination Item Recommendation Ontology) Standardized Health Status Feature
        """
        heiro_features = {
            "Demographics": [],
            "GeneralPhysicalCondition":[],
            "Lifestyle": [],
            "HealthHistory": [],
            "MentalHealth": [],
            "DiseaseRisk": [],
            "BehaviorPattern": []
        }
        
        # structured health information
        demo = self.structured_info.demographics
        heiro_features["Demographics"].extend([
            {"node_label": "Age", "hasFeatureValue": str(demo.age)},
            {"node_label": "Gender", "hasGender": demo.gender}
        ])
        if demo.marital_status:
            heiro_features["Demographics"].append({"node_label": "Marital", "hasFeatureValue": demo.marital_status})
        if demo.education_level:
            heiro_features["Demographics"].append({"node_label": "Education", "hasFeatureValue": demo.education_level})
        if demo.occupation:
            heiro_features["Demographics"].append({"node_label": "Profession", "hasFeatureValue": demo.occupation})

        gen_phy = self.structured_info.general_physical_condition
        heiro_features["GeneralPhysicalCondition"].extend([
            {"node_label": "Height", "hasFeatureValue": str(gen_phy.height)},
            {"node_label": "Weight", "hasFeatureValue": str(gen_phy.weight)},
            {"node_label": "Waist", "hasFeatureValue": str(gen_phy.waist)}
        ])
        heiro_features["GeneralPhysicalCondition"].append({"node_label": "BloodPressure", "hasFeatureValue": f"SBP:{gen_phy.blood_pressure.get("SBP")}"})
        heiro_features["GeneralPhysicalCondition"].append({"node_label": "BloodPressure", "hasFeatureValue": f"DBP:{gen_phy.blood_pressure.get("DBP")}"})
            
        life = self.structured_info.lifestyle
        heiro_features["Lifestyle"].extend([
            {"node_label": "Smoking", "hasFeatureValue": life.smoking_status},
            {"node_label": "Drinking", "hasFeatureValue": life.drinking_status},
            {"node_label": "Diet", "hasType": life.diet_pattern},
            {"node_label": "Sports", "hasFeatureValue": life.physical_activity_level},
            {"node_label": "SleepQuality", "hasFeatureValue": life.sleep_quality}
        ])
        
        med_history = self.structured_info.medical_history
        for condition in med_history.diagnosed_conditions:
            heiro_features["HealthHistory"].append({
                "node_label": "MedicalHistory", 
                "hasFeatureValue": f"Diagnosis:{condition}"
            })
        for surgery in med_history.surgeries:
            heiro_features["HealthHistory"].append({
                "node_label": "SurgeryHistory", 
                "hasFeatureValue": f"Surgery:{surgery}"
            })
        for medication in med_history.medications:
            heiro_features["HealthHistory"].append({
                "node_label": "MedicationHistory", 
                "hasFeatureValue": f"Medication:{medication}"
            })
            
        fam_history = self.structured_info.family_history
        for relative, conditions in fam_history.family_conditions.items():
            for condition in conditions:
                heiro_features["HealthHistory"].append({
                    "node_label": "FamilyHistory", 
                    "hasFeatureValue": f"{relative}:{condition}"
                })

        mental = self.structured_info.mental_status
        if mental.anxiety:
            heiro_features["MentalHealth"].append({"node_label": "Anxiety", "hasFeatureValue": str(mental.anxiety)})
        if mental.depression:
            heiro_features["MentalHealth"].append({"node_label": "Depression", "hasFeatureValue": str(mental.depression)})
        if mental.pressure:
            heiro_features["MentalHealth"].append({"node_label": "Pressure", "hasFeatureValue": str(mental.pressure)})

        # predicted disease risk
        for disease, risk_level in self.disease_risks.risk_levels.items():
            if risk_level in ["High Risk", "Medium Risk"]:
                heiro_features["DiseaseRisk"].append({
                    "node_label": "DiseaseRisk",
                    "hasFeatureValue": f"{disease}:{risk_level}"
                })

        # health behavior patterns
        engagement = self.behavior_patterns.engagement_cluster
        heiro_features["BehaviorPattern"].append({
            "node_label": "BehaviorPattern",
            "hasFeatureValue": f"Engagement:{engagement}"
        })
        
        for behavior, ratio in self.behavior_patterns.behavior_trail_distribution.items():
            if ratio > self.behavior_patterns.behavior_preference_threshold: 
                heiro_features["BehaviorPattern"].append({
                    "node_label": "BehaviorPreference",
                    "hasFeatureValue": behavior
                })

        return heiro_features