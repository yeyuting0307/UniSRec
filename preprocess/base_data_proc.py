import os
import sys
import logging
import pandas as pd
from abc import ABC, abstractmethod

# Append parent directory if needed
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

class BaseProcessor(ABC):
    def __init__(self, project_id: str, location: str, bucket_name: str,
                 dataset_name: str, stage: str):
        """
        Initialize the data processor with necessary configurations.

        :param project_id: project id.
        :param location: project location.
        :param bucket_name: bucket name.
        :param dataset_name: Dataset identifier.
        :param stage: Stage name (e.g., 'finetune', 'pretrain', 'inference').
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.dataset_name = dataset_name
        self.stage = stage

        self.data_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, 'data', stage, dataset_name
        )
        os.makedirs(self.data_dir, exist_ok=True)

    @abstractmethod
    def download_data(self, src:str) -> str:
        """
        Download the raw data file from Database (or any other source)
        and return the local file path.

        :return: Local file path to the downloaded raw data.
        """
        pass

    @abstractmethod
    def load_data(self, file_path) -> pd.DataFrame:
        """
        Load and return the raw data as a DataFrame.
        This method uses the local file path returned by download_data().

        :return: A pandas DataFrame containing raw interaction data.
        """
        pass

    @abstractmethod
    def exclude_coldstart(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exclude coldstart users (e.g., users with no interaction history).

        :param df: The input DataFrame.
        :return: A DataFrame excluding coldstart records.
        """
        pass

    @abstractmethod
    def get_coldstart_simulated_interact(self) -> pd.DataFrame:
          """
          Get simulated interaction data for coldstart users.
    
          :return: A DataFrame of simulated interaction data.
          """
          pass        
           
    def split_dataset(self, df: pd.DataFrame, 
            valid_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        Split the dataset into train, valid, and test sets.

        :param df: The input DataFrame.
        :param valid_ratio: Ratio for the validation set.
        :param test_ratio: Ratio for the test set.
        :return: Tuple of (train_df, valid_df, test_df)
        """
        n = len(df)
        train_ratio = 1 - valid_ratio - test_ratio

        train_end_idx = round(n * train_ratio)
        valid_end_idx = round(n * valid_ratio) + train_end_idx
        test_end_idx = round(n * test_ratio) + valid_end_idx

        df_train = df.iloc[:train_end_idx].sort_values(by=['user_token', 'interact_at'])
        df_valid = df.iloc[train_end_idx:valid_end_idx].sort_values(by=['user_token', 'interact_at'])
        df_test = df.iloc[valid_end_idx:test_end_idx].sort_values(by=['user_token', 'interact_at'])

        return df_train, df_valid, df_test

    @abstractmethod
    def write_inter_file(self, file_path: str, df: pd.DataFrame,
                       max_seq_length: int = 50) -> None:
        """
        Write the interaction data to a formatted file for recbole according to the
        specific stage requirements (e.g., finetune, pretrain, inference).

        :param file_path: The output file path.
        :param df: The DataFrame to write.
        :param max_seq_length: Maximum sequence length for item token sequence.
        """
        pass

    @abstractmethod
    def upload_interact_files(self, file: str, dst: str) -> None:
        """
        Upload the interaction files to Database if needed.

        :param train_file: Local path to the training interaction file.
        :param valid_file: Local path to the validation interaction file.
        :param test_file: Local path to the test interaction file.
        """
        pass

    @abstractmethod
    def create_interact(self, src: str, dst: str,
            add_simulated_interact: bool = True,
            valid_ratio: float = 0.1, test_ratio: float = 0.1,
            max_seq_length: int = 50) -> None:
        """
        Complete pipeline to create interaction data.
        Example:
          1. Load raw data.
          2. Exclude coldstart records.
          3. Split the dataset.
          4. Optionally add simulated interaction data.
          5. Write interaction files.
          6. Upload files to Database.
        """
        pass

