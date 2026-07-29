"""
Knowledge Graph Core: Health Status Profile Storage by HEIRO

@author: DanWu and Agreewithu (Ruixin Dai)
"""

from typing import Dict, Any, List
from ai_ihe_core.data_models.health_profile import HealthStatusProfile

class HeiroMapper:
    """
    Health Examination Item Recommendation Ontology Mapper
    Multidimensional Health Status Profile Standardized Mapping to HEIRO's Health Features Node
    """
    def __init__(self):
        # PEIR.owl core classes
        self.supported_feature_classes = [
            "Age", "Gender", "Marital", "Education", "Profession", # Demographics
            "MedicalHistory", "SurgeryHistory", "MedicationHistory", "FamilyHistory", # HealthHistory
            "Smoking", "Drinking", "Diet", "Sports","SleepQuality", # Lifestyle
            "Anxiety", "Depression", "Pressure", # MentalHealth
            "BloodPressure", "Height", "Weight", "Waist", # GeneralPhysicalCondition
            "DiseaseRisk", "BehaviorPattern" # Extension Health Dimension
        ]

    def map_to_ontology(self, profile: HealthStatusProfile) -> List[Dict[str, Any]]:
        """
        Convert Health Status Profile into Ontological Structured Feature List
        
        :param profile: health status profile
        :return: standardized feature list containing node types (class names) and data attributes (feature value)
        """
        heiro_features = []
        
        # demographics
        demo = profile.structured_info.demographics
        heiro_features.append({"node_label": "Age", "hasFeatureValue": str(demo.age)})
        heiro_features.append({"node_label": "Gender", "hasGender": demo.gender})
        if demo.marital_status:
            heiro_features.append({"node_label": "Marital", "hasFeatureValue": demo.marital_status})
        if demo.education_level:
            heiro_features.append({"node_label": "Education", "hasFeatureValue": demo.education_level})
        if demo.occupation:
            heiro_features.append({"node_label": "Profession", "hasFeatureValue": demo.occupation})

        # general physical condition
        gen_phy = profile.structured_info.general_physical_condition
        heiro_features.append({"node_label": "Height", "hasFeatureValue": str(gen_phy.height)})
        heiro_features.append({"node_label": "Weight", "hasFeatureValue": str(gen_phy.weight)})
        heiro_features.append({"node_label": "Waist", "hasFeatureValue": str(gen_phy.waist)})
        heiro_features.append({"node_label": "BloodPressure", "hasFeatureValue": f"SBP:{gen_phy.blood_pressure.get("SBP")}"})
        heiro_features.append({"node_label": "BloodPressure", "hasFeatureValue": f"DBP:{gen_phy.blood_pressure.get("DBP")}"})

        # lifestyle
        life = profile.structured_info.lifestyle
        heiro_features.append({"node_label": "Smoking", "hasFeatureValue": life.smoking_status})
        heiro_features.append({"node_label": "Drinking", "hasFeatureValue": life.drinking_status})
        heiro_features.append({"node_label": "Diet", "hasType": life.diet_pattern})
        heiro_features.append({"node_label": "Sports", "hasFeatureValue": life.physical_activity_level})
        heiro_features.append({"node_label": "SleepQuality", "hasFeatureValue": life.sleep_quality})
        
        # health history
        med_history = profile.structured_info.medical_history
        for condition in med_history.diagnosed_conditions:
            heiro_features.append({"node_label": "MedicalHistory", "hasFeatureValue": f"Diagnosis:{condition}"})
        for surgery in med_history.surgeries:
            heiro_features.append({"node_label": "SurgeryHistory", "hasFeatureValue": f"Surgery:{surgery}"})
        for medication in med_history.medications:
            heiro_features.append({"node_label": "MedicationHistory", "hasFeatureValue": f"Medication:{medication}"})
        
        family_hist = profile.structured_info.family_history
        for relative, conditions in family_hist.family_conditions.items():
            for condition in conditions:
                heiro_features.append({"node_label": "FamilyHistory", "hasFeatureValue": f"{relative}:{condition}"})

        # mental health
        mental = profile.structured_info.mental_status
        heiro_features.append({"node_label": "Anxiety", "hasFeatureValue": str(mental.anxiety)})
        heiro_features.append({"node_label": "Depression", "hasFeatureValue": str(mental.depression)})
        heiro_features.append({"node_label": "Pressure", "hasFeatureValue": str(mental.pressure)})

        # historical HERs disease risk
        for disease, risk_level in profile.disease_risks.risk_levels.items():
            if risk_level in ["High Risk", "Medium Risk"]:
                # generate specific risk state features dynamically
                heiro_features.append({
                    "node_label": "DiseaseRisk", 
                    "hasFeatureValue": f"{disease}:{risk_level}"
                })

        # health behavior pattern
        engagement = profile.behavior_patterns.engagement_cluster
        heiro_features.append({
            "node_label": "BehaviorPattern",
            "hasFeatureValue": f"Engagement:{engagement}"
        })

        for behavior, ratio in profile.behavior_patterns.behavior_trail_distribution.items():
            if ratio > profile.behavior_patterns.behavior_preference_threshold:
                heiro_features.append({"node_label": "BehaviorPreference", "hasFeatureValue": behavior})

        return heiro_features