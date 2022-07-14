""" This script generates artificial training labelled data to serve as input data for
all Machine Learning algorithms built for Product Matching problem """
import os
from datetime import date
import numpy as np
import pandas as pd
from string_similarity_metrics import StringMetrics
from data_preprocessing import DataPreprocessing
import settings


class SyntheticDataGeneration:
    """The class is used to create synthetic labelled data for product matching algorithms"""
    @staticmethod
    def create_synthetic_data(product_df, iterations):
        """
        Creates synthetic training data from the correctly matched data by grouping on the cluster_label column and
        reshuffling the internal_name to create data that contain incorrect matches.
        Args:
            product_df: The input dataframe containing x independent product variables
            iterations: The number of iterations / shuffling to be done to generate artificial labelled data
        Returns:
            df_output: labelled data with target variable
        """
        df_output = product_df
        i = 1
        while i <= iterations:
            # Create synthetic data by shuffling the column using a groupby
            df_s = product_df.copy()
            df_s['shuffled_internal_name'] = df_s['title']
            df_s['shuffled_internal_name'] = df_s.groupby('mas_cat_id')['title'].transform(np.random.permutation)
            # Add the correct value to the match column
            df_s['match'] = np.where(df_s['title'] == df_s['shuffled_internal_name'], 1, 0)
            # Create internal name column
            df_s['title'] = np.where(df_s['shuffled_internal_name'] != '', df_s['shuffled_internal_name'],
                                     df_s['title'])
            df_output = df_output.append(df_s)
            df_output = df_output.drop(columns=['shuffled_internal_name'])
            # iterations
            i += 1
        return df_output

    def execute(self):
        """
        This function produces training dataframe for Machine Learning Model Training by extracting additional
        features and cleaning the data.
        Returns:
            train_df: Labelled data ready for model training
         """

        master_df = pd.read_csv(f"{os.getcwd()}/data/product_master_data.csv")
        master_df = master_df.astype(str)
        print(master_df.head(3))
        # Concatenate all supplier attributes before shuffling takes place
        master_df['title'] = master_df['title'] + '%@#5' + master_df['default_price_cents'] + '%@#5' + \
                             master_df['eans'] + '%@#5' + master_df['sup_cat_id']
        # label all records as match = 1
        master_df['match'] = 1
        # shuffle internally
        train_df = self.create_synthetic_data(master_df, 5)
        train_df['title'] = train_df['title'].str.strip()
        # Separate back again after shuffling
        train_df[['title', 'default_price_cents', 'eans', 'sup_cat_id']] = train_df['title'].str.split('%@#5',
                                                                                                       expand=True)
        # Data aggregation and preprocessing
        train_df = StringMetrics(train_df).execute()
        train_df = DataPreprocessing().execute(train_df)
        return train_df


if __name__ == '__main__':
    ml_train_df = SyntheticDataGeneration().execute()
    # save locally
    ml_train_df.to_csv(settings.project_dir+f"/training_data/training_data_{date.today()}.csv", index=False)

