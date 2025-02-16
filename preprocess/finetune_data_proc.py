import os
import sys
import logging
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from preprocess.base_data_proc import BaseProcessor

class FinetuneProcessor(BaseProcessor):
    def __init__(self, project_id: str, location: str, bucket_name: str,
                    dataset_name: str, stage: str):
        super().__init__(project_id, location, bucket_name, dataset_name, stage)

    def download_data(self, src) -> str:
        ...

    def load_data(self, file_path) -> pd.DataFrame:
       ...

    def exclude_coldstart(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def write_inter_file(self, file_path, df, max_seq_length = 50):
        with open(file_path, 'w') as f:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for _, row in df.iterrows():
                user_token = row['user_token']
                item_token_series = row['item_token_series']
                item_token = row['item_token']
                combined_item = item_token_series.split(',')
                combined_item = combined_item[-max_seq_length:]
                f.write(f"{user_token}\t{' '.join(combined_item)}\t{item_token}\n")

    def upload_interact_files(self, file_path, dst):
        ...
    
    def create_interact(self, src: str, dst: str,
            add_simulated_interact: bool = True,
            valid_ratio: float = 0.1, test_ratio: float = 0.1,
            max_seq_length: int = 50):
        
        """ Create finetune data for recbole """
        # Load and preprocess data
        file_path = self.download_data(src)
        df = self.load_data(file_path)
        df = self.exclude_coldstart(df)
        df_train, df_valid, df_test = self.split_dataset(df, valid_ratio, test_ratio)

        # Add simulated interaction data if required
        if add_simulated_interact:
            df_simulated = self.get_coldstart_simulated_interact()
            df_simulated = df_simulated[['user_token', 'item_token_series', 'item_token']]
            df_simulated['interact_at'] = pd.to_datetime('now', utc=True)
            df_train = pd.concat([df_train, df_simulated], ignore_index=True)
            logging.warning(f"Added {len(df_simulated)} simulated interaction records.")

        logging.warning(f"Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}")

        # Define output file paths
        train_file = os.path.join(self.data_dir, f"{self.dataset_name}.train.inter")
        valid_file = os.path.join(self.data_dir, f"{self.dataset_name}.valid.inter")
        test_file = os.path.join(self.data_dir, f"{self.dataset_name}.test.inter")

        # Write interaction data to files
        self.write_inter_file(train_file, df_train, max_seq_length)
        self.write_inter_file(valid_file, df_valid, max_seq_length)
        self.write_inter_file(test_file, df_test, max_seq_length)

        # Upload files to Database if needed
        self.upload_interact_files(train_file, dst)
        self.upload_interact_files(valid_file, dst)
        self.upload_interact_files(test_file, dst)