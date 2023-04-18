import os 
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(Self):
        try:
            logging.info("READING THE DATA")
            df=pd.read_csv('notebook\data\winequality-red.csv')
            os.makedirs(os.path.dirname(Self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(Self.data_ingestion_config.raw_data_path)
            logging.info("INITIATING TRAIN TEST SPLIT")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=4)
            train_set.to_csv(Self.data_ingestion_config.train_data_path)
            test_set.to_csv(Self.data_ingestion_config.test_data_path)

            return (
                Self.data_ingestion_config.train_data_path,
                Self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initate_model_training(train_arr,test_arr))


