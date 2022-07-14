""" This script is used to check and gauge the performance of the rules extracted and their
impact on effectiveness of the Rule-based classifier. The evaluation metrics help
understand if the rules can be formulated better in future """
import os
import numpy as np
import pandas as pd


class RuleBasedClassification:
    """This class is used to execute Rule based classification on Supplier inventory data to classify products as match
    or not"""
    def __init__(self, supplier_id=142, is_tobacco_flag=False, rule_no=3):
        """
        initializing the class by making the input available for all the class members
        Args:
            supplier_id: id of supplier
            is_tobacco_flag: if products are tobacco / not (True/False)
            rule_no: specify the rule number and accordingly pre-condition will be applied
            sav_to_dir: save results to this dir
         """
        self.supplier_id = supplier_id
        self.is_tobacco_flag = is_tobacco_flag
        self.rule_no = rule_no

    @staticmethod
    def extract_features_for_rules(product_df) -> pd.DataFrame:
        """ This function extracts some features that is required for predefined rules such as
         max sim score amongst all candidate products etc.
        Args:
            product_df: dataframe containing pairs of products for comparison
        Returns:
            grouped_df: df with additional extracted features req for rule extraction
        """
        grouped_df = pd.DataFrame()
        # get max sim score amongst candidate products
        grouped_df['wg_sim_score_max'] = product_df.groupby(['sup_idx'], sort=True)['wg_sim_score'].max()
        grouped_df = grouped_df.reset_index()
        # get 10 price values in sorted ascending order
        grouped_df['price_knn'] = product_df.groupby('sup_idx')['price_diff'].apply(list)
        # handle null values
        grouped_df['price_knn'] = [sorted(price_list)[:10] if price_list is not np.NaN else ''
                                   for price_list in grouped_df['price_knn']]
        return grouped_df

    @staticmethod
    def extract_rule_1(product_df) -> pd.DataFrame:
        """ This function pre-defines rule number 1 and extracts match prediction according to it
        Args:
            product_df: dataframe containing pairs of products for comparison
        Returns:
            product_df: product_df with additional field for match prediction
        """
        product_df['match_pred'] = product_df.apply(lambda x: 1 if (x['wg_sim_score'] >= 0.80) & (x['ean_match_flag']
                                                                                                  == 1) else 0, axis=1)
        return product_df

    @staticmethod
    def extract_rule_2(product_df) -> pd.DataFrame:
        """ This function pre-defines rule number 2 and extracts match prediction according to it
        Args:
            product_df: dataframe containing pairs of products for comparison
        Returns:
            product_df: product_df with additional field for match prediction
        """
        product_df['match_pred'] = product_df.apply(lambda x: 1 if (x['wg_sim_score'] == x['wg_sim_score_max']) &
                                                                       (x['price_diff'] in x['price_knn']) else 0,
                                                    axis=1)
        return product_df

    @staticmethod
    def extract_rule_3(product_df) -> pd.DataFrame:
        """ This function pre-defines rule number 3 and extracts match prediction according to it
        Args:
            product_df: dataframe containing pairs of products for comparison
        Returns:
            product_df: product_df with additional field for match prediction
        """
        product_df['match_pred'] = product_df.apply(lambda x: 1 if (x['wg_sim_score'] == x['wg_sim_score_max']) else 0,
                                                    axis=1)
        return product_df

    def get_rule_based_classification(self, product_df: pd.DataFrame) -> pd.DataFrame:
        """ This function extracts any 1 of the 3 rules defined based on the parameter passed by the user
        Args:
            product_df: dataframe containing pairs of products for comparison
        Returns:
            product_df: resulting df after classification
        """
        lookup_df = self.extract_features_for_rules(product_df)
        product_df = pd.merge(product_df, lookup_df, on=['sup_idx', 'sup_idx'])
        if self.rule_no == 1:
            self.extract_rule_1(product_df)
        elif self.rule_no == 2:
            self.extract_rule_2(product_df)
        elif self.rule_no == 3:
            self.extract_rule_3(product_df)
        return product_df


