"""
HealthBehavior Core: Health Behavior Trail Markov Modeling

@author: DanWu and Agreewithu (Ruixin Dai)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

class MarkovBehaviorModel:
    """
    Behavioral Pattern Extraction Module Based on Markov Chain
    Session Construction, State Transition Probability Matrices, and Calculation of Multi-dimensional Behavioral Trajectory Proportions
    """
    def __init__(self, session_timeout_seconds: int = 600):
        # session spliting
        self.session_timeout_seconds = session_timeout_seconds
        self.transition_counts = defaultdict(lambda: defaultdict(int))

    def _add_sequence(self, sequence: List[str]):
        """Iter Sequence"""
        for i in range(len(sequence) - 1):
            self.transition_counts[sequence[i]][sequence[i + 1]] += 1

    def build_markov_chain(self, raw_logs: pd.DataFrame) -> pd.DataFrame:
        """
        Construct Markov Chain
        Logs: ['userId', 'event_target', 'timestamp']
        """
        raw_logs['timestamp'] = pd.to_datetime(raw_logs['timestamp'])
        sorted_data = raw_logs.sort_values(by=['userId', 'timestamp']).reset_index(drop=True)
        
        sessions_vectors = []
        sessions_cntr = 0

        for user in np.unique(sorted_data['userId']):
            user_logs = sorted_data[sorted_data['userId'] == user].reset_index(drop=True)
            current_session = [('start', 'nan')]

            if len(user_logs) == 1:
                current_session.append((user_logs.iloc[0]['event_target'], user_logs.iloc[0]['timestamp']))
                current_session[0] = ('start', current_session[1][1])
                current_session.append(('exit', current_session[-1][1]))
                for k in range(len(current_session) - 1):
                    sessions_vectors.append([user, current_session[k][0], current_session[k + 1][0], sessions_cntr, current_session[k][1]])
                sessions_cntr += 1
                self._add_sequence([x[0] for x in current_session])
                continue

            for i in range(len(user_logs) - 1):
                current_session.append((user_logs.iloc[i]['event_target'], user_logs.iloc[i]['timestamp']))
                time_diff = user_logs.iloc[i + 1]['timestamp'] - user_logs.iloc[i]['timestamp']
                
                # out of session_timeout_seconds
                if time_diff.total_seconds() >= self.session_timeout_seconds or i == len(user_logs) - 2:
                    if i == len(user_logs) - 2 and time_diff.total_seconds() < self.session_timeout_seconds:
                         current_session.append((user_logs.iloc[i + 1]['event_target'], user_logs.iloc[i + 1]['timestamp']))
                    
                    current_session[0] = ('start', current_session[1][1])
                    current_session.append(('exit', current_session[-1][1]))
                    
                    for k in range(len(current_session) - 1):
                        sessions_vectors.append([user, current_session[k][0], current_session[k + 1][0], sessions_cntr, current_session[k][1]])
                    
                    sessions_cntr += 1
                    self._add_sequence([x[0] for x in current_session])
                    current_session = [('start', 'nan')]
                    
                    if i == len(user_logs) - 2 and time_diff.total_seconds() >= self.session_timeout_seconds:
                         current_session.append((user_logs.iloc[i + 1]['event_target'], user_logs.iloc[i + 1]['timestamp']))
                         current_session[0] = ('start', current_session[1][1])
                         current_session.append(('exit', current_session[-1][1]))
                         for k in range(len(current_session) - 1):
                             sessions_vectors.append([user, current_session[k][0], current_session[k + 1][0], sessions_cntr, current_session[k][1]])
                         sessions_cntr += 1
                         self._add_sequence([x[0] for x in current_session])

        sessions_df = pd.DataFrame(sessions_vectors, columns=['userId', 'from', 'to', 'session_id', 'timestamp'])
        return sessions_df

    def get_transition_matrix(self) -> defaultdict:
        """Transition Probability Matrix and Normalization"""
        transition_matrix = defaultdict(lambda: defaultdict(float))
        for from_page in self.transition_counts:
            total = sum(self.transition_counts[from_page].values())
            for to_page in self.transition_counts[from_page]:
                transition_matrix[from_page][to_page] = self.transition_counts[from_page][to_page] / total
        return transition_matrix

    def calculate_behavior_trail_proportions(self, sessions_df: pd.DataFrame, state_group_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Multidimensional Behavior Trajectory Preference Distribution of Each User
        
        :param sessions_df: build_markov_chain output session dataframe
        :param state_group_mapping: state behavior mapping
               e.g. {'Task': ['p_TaskBP', 'p_TaskDrug'], 'Report': ['p_HealthReport']}
        """
        sessions_df['num'] = 1
        count_state = sessions_df.groupby(['userId', 'from'], as_index=False)[['num']].count()
        
        patient_trails = []
        categories = list(state_group_mapping.keys())
        
        for user_id in np.unique(count_state['userId']):
            user_data = count_state[count_state['userId'] == user_id]
            group_counts = {cat: 0 for cat in categories}
            
            for state in np.unique(user_data['from']):
                user_state = user_data[user_data['from'] == state]
                state_count = user_state.iloc[0]['num']
                
                for cat, states in state_group_mapping.items():
                    if state in states:
                        group_counts[cat] += state_count
                        break
            
            total_sum = sum(group_counts.values())
            proportions = {cat: (count / total_sum if total_sum != 0 else 0) for cat, count in group_counts.items()}
            
            row = [user_id] + [proportions[cat] for cat in categories]
            patient_trails.append(row)
            
        columns = ['userId'] + [f"{cat}_Ratio" for cat in categories]
        df_trail_proportions = pd.DataFrame(patient_trails, columns=columns)
        return df_trail_proportions