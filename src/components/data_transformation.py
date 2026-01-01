import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            num_cols = ["writing score", "reading score"]
            cat_cols = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

            logging.info("Numerical columns standardizing completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    # ADD sparse_output=False HERE to prevent the sparse matrix error
                    ("one_hot_encode", OneHotEncoder(sparse_output=False)), 
                    ("scaler", StandardScaler(with_mean=False)) # with_mean=False is safer for encoded data
                ]
            )

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer([
                ("num_trans", num_pipeline, num_cols),
                ("cat_trans", cat_pipeline, cat_cols)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading train&test data is completed")
            logging.info("obtaining preprocessing object")
            preprocessor_obj=self.get_data_transformation_object()
            target_column="math score"
            num_cols=["writing score","reading score"]
            input_feature_train_df=train_df.drop([target_column],axis=1)
            target_feature_train=train_df[target_column]
            
            input_feature_test_df=test_df.drop([target_column],axis=1)
            target_feature_test=test_df[target_column]
            logging.info("Applying preprocessing object on training and test data")
            input_feature_train_array=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_df)
            train_arr=np.c_[
                input_feature_train_array, np.array(target_feature_train)
            ]
            test_arr=np.c_[
                input_feature_test_array,target_feature_test
            ]
            logging.info("saving preprocessing objects")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)