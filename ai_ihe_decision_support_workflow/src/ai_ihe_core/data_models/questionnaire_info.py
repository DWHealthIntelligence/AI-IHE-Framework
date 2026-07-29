"""
Health Status Descirption: Questionnaire (Structual data ruled by ontology)

@author: DanWu and Agreewithu (Ruixin Dai)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Demographics:
    """Demographic Basic Information"""
    age: int
    gender: str
    marital_status: Optional[str] = None
    education_level: Optional[str] = None
    occupation: Optional[str] = None

@dataclass
class GeneralPhysicalCondition:
    """General Physical Condition"""
    height: float
    weight: float
    waist: float
    blood_pressure: Dict[str, float]

@dataclass
class MedicalHistory:
    """Disease History"""
    # diagnosed diseases (e.g. ['Hypertension', 'Hyperlipidemia'])
    diagnosed_conditions: List[str] = field(default_factory=list)
    # surgeries
    surgeries: List[str] = field(default_factory=list)
    # medications: drug names
    medications: List[str] = field(default_factory=list)

@dataclass
class Lifestyle:
    """Lifestyle"""
    smoking_status: str  # smoking: 'Never', 'Former', 'Current'
    drinking_status: str # drinking: 'Never', 'Occasional', 'Frequent'
    diet_pattern: str    # diet: 'Balanced', 'High-Salt', 'High-Fat'
    physical_activity_level: str # activity lev: 'Low', 'Moderate', 'High'
    sleep_quality: Optional[str] = None

@dataclass
class FamilyHistory:
    """Genetic History"""
    # Medical history of chronic diseases in direct relatives of the family (e.g. {'Father': ['Type 2 Diabetes'], 'Mother': ['Hypertension']})
    family_conditions: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class MentalHealth:
    """Mental Health"""
    anxiety: bool = False # 'Anxiety'
    depression: bool = False # 'Depression'
    pressure: bool = False # 'Pressure'

@dataclass
class HealthQuestionnaire:
    """
    Complete Structured Health Survey Questionnaire Information
    """
    participant_id: str
    demographics: Demographics
    general_physical_condition: GeneralPhysicalCondition
    medical_history: MedicalHistory
    lifestyle: Lifestyle
    family_history: FamilyHistory
    mental_status: MentalHealth