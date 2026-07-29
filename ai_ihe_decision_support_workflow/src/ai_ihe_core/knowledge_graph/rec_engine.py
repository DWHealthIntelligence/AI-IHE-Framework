"""
Knowledge Graph Core: Health Examination Item Recommendation Engine

@author: DanWu and Agreewithu (Ruixin Dai)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from neo4j import GraphDatabase

class Neo4jConnectionManager:
    """
    Neo4j Connnection Manager
    Manage the Connection Lifecycle and Session Transactions of Property Graph Database
    """
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """close conn"""
        if self.driver:
            self.driver.close()

    def execute_read(self, query_func, *args, **kwargs) -> Any:
        """
        exec session transactions
        
        :param query_func: callback session
        """
        with self.driver.session() as session:
            return session.execute_read(query_func, *args, **kwargs)


class AbstractRecommendationStrategy(ABC):
    """
    Abstract Interface for Specific Reasoning Strategy
    Rule Driving Query for the Health Status Profile of HEIRO
    Realized by all Decision Support Web Service Layers
    """
    
    @abstractmethod
    def query_recommendations(self, tx, heiro_features: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """
        Executing Recommendation Queries in Graph Database Transactions
        
        :param tx: Neo4j transactions
        :param heiro_features: HEIRO standardized feature dictionary
                               (ref of: HealthStatusProfile.to_heiro_features())
        :return: return the recommended result list, including the recommended items and their corresponding reasoning evidence
                 expect format: 
                 [
                     {
                         "item": "Lung_CT", 
                         "reason": "[Smoking: Current] -> recommends -> [Lung_CT]", 
                         "type": "Instrument"
                     }, ...
                 ]
        """


class ExaminationItemRecommendationSystem:
    """
    AI-assisted Health Examination Item Recommendation Coordinator
    """
    def __init__(self, db_manager: Neo4jConnectionManager, strategy: AbstractRecommendationStrategy):
        self.db_manager = db_manager
        self.strategy = strategy

    def generate_strategy(self, heiro_features: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Utilize the Property Knowledge Graph to Deduce Health Check-up Recommendation Strategies Through Standardized User Health Status Features (Defined by HEIRO)
        
        :param heiro_features: HealthStatusProfile.to_heiro_features()
        :return: return the recommended result list, including the recommended items and their corresponding reasoning evidence
        """

        recommended_strategy = {
            "physique_items": set(),
            "laboratory_items": set(),
            "instrument_items": set(),
            "reasoning_paths": []  # record reasoning path for healthcare provider check and revise
        }

        
        raw_recommendations = self.db_manager.execute_read(
            self.strategy.query_recommendations, 
            heiro_features
        )

        if raw_recommendations:
            for rec in raw_recommendations:
                item_name = rec.get("item")
                item_type = str(rec.get("type", "Physique")).lower()
                reason = rec.get("reason")
                
                if not item_name:
                    continue

                if reason and reason not in recommended_strategy["reasoning_paths"]:
                    recommended_strategy["reasoning_paths"].append(reason)
                    
                if "instrument" in item_type:
                    recommended_strategy["instrument_items"].add(item_name)
                elif "laboratory" in item_type:
                    recommended_strategy["laboratory_items"].add(item_name)
                else:
                    recommended_strategy["physique_items"].add(item_name)

        recommended_strategy["physique_items"] = list(recommended_strategy["physique_items"])
        recommended_strategy["laboratory_items"] = list(recommended_strategy["laboratory_items"])
        recommended_strategy["instrument_items"] = list(recommended_strategy["instrument_items"])

        return recommended_strategy