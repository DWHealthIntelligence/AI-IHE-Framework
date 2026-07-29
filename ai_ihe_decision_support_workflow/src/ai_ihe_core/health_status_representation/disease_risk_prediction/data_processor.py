"""
Disease Risk Prediction Core: Deep Learning Data Processor

@author: DanWu and Agreewithu (Ruixin Dai)
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Any

class HistoricalRecordProcessor:
    """
    Historical Health Examination Records Serialization Processor
    HERs -> (delta_t, feature, missing indicator)
    """
    
    def add_indicator(self, x: pd.DataFrame) -> pd.DataFrame:
        """ Physical Examination Indicator Process (Indicator Rely on Calculation)"""
        indicator_list = ['Height', 'Pluse', 'Waist', 'Hip', 'sbp', 'dbp']
        features = x
        for feature in np.array(indicator_list):
            if feature == 'Height':
                features['BMI'] = x['Weight'] / (x['Height'] * x['Height'])
                BMI_value = features[['BMI']]
                indicator = []
                for i in range(len(BMI_value)):
                    num = BMI_value.iloc[i,0]
                    
                    if np.isnan(num):
                        indicator.append(np.nan)
                    else:
                        indicator.append(self.cal_indicator(num, 'BMI'))
                        
                features['BMI_indicator'] = indicator
            
            elif feature == 'Waist':
                feature_records = x[[feature]]
                indicator = []
                for i in range(len(feature_records)):
                    num = feature_records.iloc[i,0]
                    gender = x.loc[x['PE_ID'] == features.iloc[i]['PE_ID'], ['SEX']].iat[0,0]

                    if np.isnan(num):
                        indicator.append(np.nan)
                    else:
                        indicator.append(self.cal_indicator1(num, feature, gender))
                        
                features['%s_indicator'%feature] = indicator
                
            elif feature == 'Hip':
                features['WH'] = x['Waist'] / x['Hip']
                WH_value = features[['WH']]
                indicator = []
                for i in range(len(WH_value)):
                    num = WH_value.iloc[i,0]
                    gender = x.loc[x['PE_ID'] == features.iloc[i]['PE_ID'], ['SEX']].iat[0,0]

                    if np.isnan(num):
                        indicator.append(np.nan)
                    else:
                        indicator.append(self.cal_indicator1(num, 'WH', gender))
                        
                features['WH_indicator'] = indicator 
            else:
                feature_records = x[[feature]]
                indicator = []
                for i in range(len(feature_records)):
                    num = feature_records.iloc[i,0]
                    
                    if np.isnan(num):
                        indicator.append(np.nan)
                    else:
                        indicator.append(self.cal_indicator(num, feature))
                        
                features['%s_indicator'%feature] = indicator
        
        return features

    # HERs -> Time Series
    def calculate_delta_t(self, time_series: List[List[str]]) -> List[List[Any]]:
        """Calculate the Time Interval"""
        delta_t = []
        for t in time_series:
            dt = []
            for i in range(len(t)):
                if i == 0:
                    dt.append(0)
                else:
                    days_diff = (datetime.strptime(str(t[i]), '%Y-%m-%d') - 
                                 datetime.strptime(str(t[i-1]), '%Y-%m-%d')).days
                    if days_diff > 0:
                        dt.append(int(days_diff / 30))
                    else:
                        dt.append('#')
            delta_t.append(dt)
        return delta_t

    def get_missing_indicator(self, all_feature: List[List[List[float]]]) -> List[List[List[int]]]:
        """Missing Indicator Matrix"""
        all_missingdata_indicator = []
        for temp_f in all_feature:
            md_f = []
            for patient_feature in temp_f:
                md = [0 if np.isnan(x) else 1 for x in patient_feature]
                md_f.append(md)
            all_missingdata_indicator.append(md_f)
        return all_missingdata_indicator

    def impute_features(self, all_feature: List[List[List[float]]]) -> List[List[List[float]]]:
        """Missing Value Imputation: Forward Fill"""
        all_feature_impute = []
        for temp_f in all_feature:
            feature_impute = []
            for patient_feature in temp_f:
                impute_value = []
                # first not na value and its index
                first_value = next(sub for sub in patient_feature if not np.isnan(sub)) 
                first_value_index = patient_feature.index(first_value)
                
                for i in range(len(patient_feature)):
                    if i < first_value_index:
                        impute_value.append(first_value)
                    elif np.isnan(patient_feature[i]):
                        impute_value.append(impute_value[i-1])
                    else:
                        impute_value.append(patient_feature[i])
                feature_impute.append(impute_value)
            all_feature_impute.append(feature_impute)
        return all_feature_impute

    def process_time_series(self, raw_historical_records: pd.DataFrame) -> Dict[str, Any]:
        """
        Raw HERs to Time Series Pipeline
        """

        def _patient_list(feature_records: pd.DataFrame, feature_name:str) -> List[List[any]]:
            """DataFrame to List According Feature Name"""
            feature_records = feature_records.sort_values(by=['PE_ID', 'PE_VISIT_NO'])

            features = []
            for patient in np.unique(feature_records['PE_ID']):
                feature = feature_records.loc[feature_records['PE_ID'] == patient, [feature_name]]
                features.append(list(feature.iloc[:,0]))

            return features

        def _feature_list(records: pd.DataFrame) -> List[List[List[float]]]:
            """DataFrame to List All Features"""
            columns_name = list(records.drop(columns = ['PE_ID', 'PE_VISIT_NO', 'PE_DATE_TIME']).columns)

            all_feature = []
            for i in range(0, len(columns_name)):
                feature_name = columns_name[i]
                feature_records = records[['PE_ID', 'PE_VISIT_NO', feature_name]]
                all_feature.append(_patient_list(feature_records, feature_name))
                print(i) # for check feature num

            return all_feature

        # raw historical records add indicator
        df_indicator = self.add_indicator(raw_historical_records.iloc[:, :12])

        # re-check and re-order
        df_indicator = df_indicator[['PE_ID', 'PE_VISIT_NO', 'PE_DATE_TIME', 'AGE',	'SEX', 'Height', 'Weight', 'BMI', 'BMI_indicator', 'Pluse', 'Pluse_indicator', 'Waist', 'Waist_indicator', 'Hip', 'WH', 'WH_indicator', 'sbp', 'sbp_indicator', 'dbp', 'dbp_indicator']]

        # complete records
        df = pd.concat([df_indicator, raw_historical_records.iloc[:, 12:]], axis = 1)

        # dataframe to list
        time = _patient_list(df[['PE_ID', 'PE_VISIT_NO', 'PE_DATE_TIME']], 'PE_DATE_TIME')
        all_feature = _feature_list(df)

        delta_t = self.calculate_delta_t(time)
        missing_indicators = self.get_missing_indicator(all_feature)
        features_imputed = self.impute_features(all_feature)
                
        processed_data = {
            "features_imputed": features_imputed, # self.impute_features(...) 
            "delta_t": delta_t,          # self.calculate_delta_t(...)
            "missing_indicators": missing_indicators # self.get_missing_indicator(...)
        }
        return processed_data