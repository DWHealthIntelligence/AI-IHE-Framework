"""
AI-assist Individual Health Examination Decision Support Workflow

@author: DanWu and Agreewithu (Ruixin Dai)
"""

from typing import Dict, Any, List
from ai_ihe_core.data_models.questionnaire_info import HealthQuestionnaire
from ai_ihe_core.data_models.health_profile import HealthStatusProfile, DiseaseRiskProfile, BehaviorPatternProfile
from ai_ihe_core.health_status_representation.disease_risk_prediction.data_processor import HistoricalRecordProcessor
from ai_ihe_core.health_status_representation.disease_risk_prediction.dl_risk_model import DeepLearningRiskModel
from ai_ihe_core.health_status_representation.mhealth_behavior_analysis.engagement_ts_clustering import TimeSeriesClusterer
from ai_ihe_core.health_status_representation.mhealth_behavior_analysis.markov_behavior_trail import MarkovBehaviorModel
from ai_ihe_core.knowledge_graph.rec_engine import ExaminationItemRecommendationSystem

class IndividualizedHealthExaminationPipeline:
    """
    Individualized Health Examination Items Recommendation Pipeline
    """
    def __init__(self, 
                 dl_risk_models: List[DeepLearningRiskModel],
                 ts_clusterer: TimeSeriesClusterer,
                 markov_model: MarkovBehaviorModel,
                 recommendation_system: ExaminationItemRecommendationSystem):
        # recommendation components
        self.dl_risk_models = dl_risk_models
        self.ts_clusterer = ts_clusterer
        self.markov_model = markov_model
        self.recommendation_system = recommendation_system

    def execute_workflow(self, 
                         participant_id: str,
                         questionnaire: HealthQuestionnaire,
                         historical_records: Any,
                         mhealth_engagement_series: List[float],
                         mhealth_session_logs: List[Dict[str, str]],
                         behavior_state_mapping: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Execution of AI-assist Individual Health Examination Decision Support Workflow
        """
        # ==========================================
        # Step 1: Patient Health Profile to Health Status Profile (by Health Status Representation)
        # ==========================================
        
        # diseases risk prediction
        hers_processed_data = HistoricalRecordProcessor().process_time_series(historical_records)
        
        disease_risk_scores_dict, disease_risk_levels_dict = {}, {}
        for dl_risk_model in self.dl_risk_models:
            # disease risk score and level
            disease_risk_score, disease_risk_level = dl_risk_model.predict_risk(hers_processed_data)

            disease_risk_scores_dict[dl_risk_model.target_disease] = disease_risk_score
            disease_risk_levels_dict[dl_risk_model.target_disease] = disease_risk_level

        disease_profile = DiseaseRiskProfile(
            risk_scores=disease_risk_scores_dict,
            risk_levels=disease_risk_levels_dict
        )

        # health behavior pattern
        engagement_cluster = self.ts_clusterer.cluster_engagement(mhealth_engagement_series)
        # dynamically maintaining a first-order Markov model
        session_df = self.markov_model.build_markov_chain(mhealth_session_logs)
        trail_proportions_df = self.markov_model.calculate_behavior_trail_proportions(session_df, behavior_state_mapping)
        
        # current behavioral preference distribution
        user_trail_dict = {}
        if not trail_proportions_df.empty:
            # participant id ordered in the first line
            row = trail_proportions_df.iloc[0]
            for cat in behavior_state_mapping.keys():
                user_trail_dict[cat] = float(row.get(f"{cat}_Ratio", 0.0))

        behavior_profile = BehaviorPatternProfile(
            engagement_cluster=engagement_cluster,
            behavior_trail_distribution=user_trail_dict,
            behavior_preference_threshold = 0.12, # setting by analysis of "Multidimensional Behavior Trail and Behavior Preference" Research 
            transition_matrix = self.markov_model.get_transition_matrix()
        )

        # health profile
        health_profile = HealthStatusProfile(
            participant_id=participant_id,
            structured_info=questionnaire,
            disease_risks=disease_profile,
            behavior_patterns=behavior_profile
        )

        # ==========================================
        # Step 2: Patient Health Status Profile to Health Examination Item Recmmendation Ontology Standard Structure (for Storage and Reasoning)
        # ==========================================
        heiro_features = health_profile.to_heiro_features()

        # ==========================================
        # Step 3: Patient Health Status Profile (in HEIRO Structure) to Examination Item Recmmendation by Knowledge Graph Reasoning
        # ==========================================
        final_strategy = self.recommendation_system.generate_strategy(heiro_features)

        # return paticipant identification, examination strategy and health profile
        return {
            "participant_id": participant_id,
            "generated_strategy": final_strategy,
            "health_profile_snapshot": heiro_features
        }
