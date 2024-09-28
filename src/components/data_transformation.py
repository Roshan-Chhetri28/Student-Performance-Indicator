import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        ''' This function is responsible for Data Transformation'''
        try:
            cat_col = ['gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course']
            num_col = ['math_score', 'reading_score', 'writing_score']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median'))
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')) 
                    ('oneHotEncoder', OneHotEncoder()),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info('Numerical column encoding complete')
            logging.info('categorical column encoding complete')

            preprocessor = ColumnTransformer(
                ('num_pipeline', num_pipeline, num_col)
                ('cat_pipeline', cat_pipeline, cat_col)
            )

            return preprocessor
        except Exception as e:
            raise(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading of Test and Train data completed')

            logging.info('obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_obj()

            target_column = ['math_score']
            
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('applying preprocessing on training dataframe and testing dataframe')


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_train_df)

            train_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object: ")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except:
            pass