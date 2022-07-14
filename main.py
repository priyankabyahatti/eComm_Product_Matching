""" This script executes Rule Based Classifier and ML Based Classifier on
 Supplier Inventory file and fetches Product Matching results of both algorithms """
import argparse
import os

import pandas as pd
import numpy as np
from string_similarity_metrics import StringMetrics
from blocking import HybridBlocking
from rule_based_product_matching import RuleBasedClassification
import pickle
from data_preprocessing import DataPreprocessing
from ml_product_matching_modelling import MLClassification
import time
import settings


def get_args():
    """ This is a helper function used to initialize and parse arguments need in the Class Product Matching.
     Returns:
         arguments: arguments passed from user via terminal
    """
    parser = argparse.ArgumentParser(description='Product Matching Tool')
    # required arguments
    parser.add_argument('--supplier_file', type=str, help='Enter path of supplier file you wish import',
                        required=True)
    # optional argument
    parser.add_argument('--method', nargs='?', default='both', help='Enter the method with which you want to run PM'
                                                                    'rule-based/ml-based')
    arguments = parser.parse_args()
    return arguments


class ProductMatching:
    """This module executes both Rule-based and Machine Learning based methods for the
    given Supplier inventory file"""

    def __init__(self, arguments):
        # Class is initialized with mandatory and optional args
        self.args = arguments

    @staticmethod
    def read_inventory_data(data_file: str) -> pd.DataFrame:
        """ This function reads supplier inventory file to a Pandas dataframe
        Args:
            data_file: csv, excel etc files
        Returns:
            inventory_df: supplier inventory as pandas df
        """
        inventory_df = pd.DataFrame()
        if data_file.lower().endswith('.csv'):
            inventory_df = pd.read_csv(data_file)
        if data_file.endswith('.xlsx'):
            inventory_df = pd.read_excel(data_file, engine='openpyxl')
        if data_file.endswith('.xls'):
            inventory_df = pd.read_excel(data_file)
        if data_file.endswith('.ods'):
            inventory_df = pd.read_excel(data_file, engine="odf")
        return inventory_df

    @staticmethod
    def process_x_train_data(processed_df: pd.DataFrame) -> np.ndarray:
        """ This function processes and standardizes X train dataset
        Args:
            processed_df: train (x independent features) dataset for classification
        Returns:
            X_unseen: standardised independent features not seen so far (without target labels)
        """
        # filter important features for classification
        x_new = processed_df[['wg_sim_score', 'price_diff', 'ean_match_flag', 'size_diff']].values
        # perform scaling to bring all features on single unit
        x_scaled = MLClassification().data_standardisation(x_new)
        # Replace NaN with zero and infinity with large finite numbers
        x_unseen = np.nan_to_num(x_scaled)
        return x_unseen

    @staticmethod
    def get_ml_prediction_results(x_unseen) -> list:
        """ This function loads saved pickled ml model and uses for predicting on unseen data
        Args:
            x_unseen: unseen data to be predicted 1 or 0
            results_dir: load ml model from this dir
        Returns:
            matched_result_list: list of matched predictions
        """
        filename = settings.project_dir+"/ml_models/XG_Boost.pkl"
        # load the model
        model_loaded = pickle.load(open(filename, "rb"))
        # get prediction for sample
        matched_result_list = model_loaded.predict(x_unseen)
        return matched_result_list

    @staticmethod
    def get_top_n_results(matched_df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        This function only gets top 3 ranked records based on the similarity score in descending order
        Args:
            matched_df: input dataframe containing matched results (match = 1 or 0)
            n: the number of results to retrieve
        Returns:
            top_n_df: dataframe containing match = 1 results with at most 3 match candidates for every product
        """
        # subset matched values
        temp_df = matched_df[matched_df['match_pred'] == 1]
        # sort by supplier id in ascending order, and sim score in descending order and then select top n
        top_n_df = temp_df.sort_values(['sup_idx', 'wg_sim_score'], ascending=[True, False]).groupby('sup_idx').head(n)
        # filter only req columns
        top_n_df = top_n_df[['sup_idx', 'title', 'mas_title', 'p_id', 'cp_id']].sort_values('sup_idx', ascending=True)
        return top_n_df

    def save_matching_results(self, processed_df: pd.DataFrame, method: str) -> None:
        """ This function saves pickled ml model and uses for predicting on unseen data
        Args:
            processed_df: unseen data to be predicted 1 or 0
            results_dir: load ml model from this dir
            method: type of method to be selected (rule-based, or ml-based)
        """
        # save ml results
        processed_df.to_csv(settings.project_dir+f"/results/{method}.csv", index=False)
        # get max top 3 results for each product
        top_3_results_df = self.get_top_n_results(processed_df, 3)
        # save auto-match results
        top_3_results_df.to_csv(settings.project_dir+f"/results/{method}_automatch.csv", index=False)

    def get_rule_based_results(self, processed_df: pd.DataFrame) -> None:
        """ This function gets rule based results and saves the results locally
        Args:
            processed_df: pre-processed data for classification
            results_dir: save the file to dir
        """
        rule_based_results_df = RuleBasedClassification().get_rule_based_classification(processed_df)
        # save results
        self.save_matching_results(rule_based_results_df, "rule-based-results")

    def get_ml_based_results(self, processed_df: pd.DataFrame) -> None:
        """ This function gets ml based results and saves the results locally
        Args:
            processed_df: pre-processed data for classification
            results_dir: path to save results
        Returns:
            filepath: save the file to dir
        """
        X_unseen = self.process_x_train_data(processed_df)
        processed_df['match_pred'] = self.get_ml_prediction_results(X_unseen)
        # save ml results
        self.save_matching_results(processed_df, "ml_based_results")

    def select_product_matching_method(self, processed_df: pd.DataFrame, method: str) -> None:
        """
        This function gives the ability to select only one particular type of method (rule or ml) at a time.
        The user can however use select both methods
           Args:
               processed_df: pre-processed data which is ready classification
               method: rule-based or ml-based
         """
        if method == 'rule_based':
            # Get Rule Based Classification Results
            start = time.time()
            self.get_rule_based_results(processed_df)
            end = time.time()
            print(f"Generated {method} Matched Results.")
            print(f"Total Time Taken: {method} ", end - start)

        elif method == 'ml_based':
            # Get ML Based Classification Results
            start = time.time()
            self.get_ml_based_results(processed_df)
            end = time.time()
            print(f"Generated {method} Matched Results.")
            print(f"Total Time Taken: {method} ", end - start)

        else:
            # both
            start = time.time()
            self.get_ml_based_results(processed_df)
            self.get_rule_based_results(processed_df)
            end = time.time()
            print(f"Generated {method} Matched Results.")
            print(f"Total Time Taken: {method} ", end - start)

    def execute(self):
        """
        This execute function executes Product matching algorithm to classify if products are match or not.
        """
        settings.create_required_dirs()
        dp = DataPreprocessing()
        # Read master data
        master_df = pd.read_csv(f"{os.getcwd()}/data/product_master_data.csv", dtype={"p_eans":"str",
                                                                                      "cp_eans":"str",
                                                                                      "cp_un_eans":"str"})
        master_df = master_df.astype(str)
        columns = ['title', 'default_price_cents', 'eans', 'sup_cat_id']
        master_df.drop(columns, inplace=True, axis=1)
        # get supplier inventory data
        supplier_df = self.read_inventory_data(self.args.supplier_file)
        # create product pair comparisons
        candidate_pair_list = HybridBlocking(supplier_df, master_df, "title", "mas_title").tfidf_based_blocking()
        candidate_pair_df = dp.merge_blocked_candidates_to_df(candidate_pair_list, supplier_df, master_df)
        # calculate string metrics
        sim_cores_df = StringMetrics(candidate_pair_df).execute()
        # execute data preprocessing
        processed_df = dp.execute(sim_cores_df, is_tobacco_flag=False)
        # user can activate any one or all method type results
        self.select_product_matching_method(processed_df, self.args.method)


if __name__ == "__main__":
    # get arguments
    args = get_args()
    # execute product matching
    ProductMatching(args).execute()
