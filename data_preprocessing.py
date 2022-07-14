""" This module deals with high level data preprocessing required for rule-based and ml-based classifiers """
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, \
    classification_report
import extract_from_title


class DataPreprocessing:
    @staticmethod
    def calculate_performance_evaluation(y_true, y_pred, classifier_type) -> None:
        """
        This function evaluates & prints the results got from the Rule-Based classification by calculating
        some performance metrics
        Args:
            y_true: list of actual target values
            y_pred: list of predicted target values
            classifier_type: type of approach used
        """
        metric_dict = {"\nPrecision score:": precision_score(y_true, y_pred),
                       "\nRecall score:": recall_score(y_true, y_pred),
                       "\nAccuracy score:": accuracy_score(y_true, y_pred),
                       "\nF1 Score: ": f1_score(y_true, y_pred),
                       "\nConfusion Matrix:": confusion_matrix(y_true, y_pred),
                       "\nClassification Report": classification_report(y_true, y_pred)}
        print("\n===================================================")
        print(f"Performance of {classifier_type} ")
        print("===================================================")
        for key, value in metric_dict.items():
            print(key, value)

    @staticmethod
    def merge_blocked_candidates_to_df(candidates, source_df, target_df) -> pd.DataFrame:
        """
        This function reconstructs the entire dataframe using the multi index candidates pairs by merging with
        supplier and master df
        Args:
            candidates = candidate pairs generated using blocking
            source_df = supplier dataframe
            target_df = master dataframe
        Returns:
            candidates_df = merged dataframe from supplier and master dfs
        """
        supplier_idx_list, product_idx_list = zip(*candidates)
        candidates_df = pd.DataFrame({'sup_idx': supplier_idx_list, 'mas_idx': product_idx_list})
        candidates_df = candidates_df.merge(source_df, left_on='sup_idx', right_on=source_df.index)\
            .merge(target_df, left_on='mas_idx', right_on=target_df.index)
        return candidates_df

    @staticmethod
    def process_ean_columns(product_df) -> pd.DataFrame:
        """ This function fixes some issues with ean column from master data such as removing whitespaces,
        and suppressing scientific notation when sometimes caused from excel
        Args:
            product_df: dataframe containing ean column
        Returns:
            product_df: processed dataframe with ean column"""
        # converting to string suppresses scientific notation
        product_df['eans'] = product_df['eans'].astype(str).replace(',', ' ', regex=True)
        # remove trailing and leading whitespace
        product_df['eans'] = [ean.rstrip().lstrip() for ean in product_df['eans']]
        # detect values such as 36873384787.0 and replace .0 with space
        product_df['eans'] = product_df['eans'].replace(r'\.0$', '', regex=True)
        # convert column values to lists
        product_df['eans'] = product_df.eans.apply(lambda x: x.split(' '))
        return product_df

    @staticmethod
    def extract_brand_from_product_title(title_column: pd.Series) -> str:
        """
        This function extracts brand feature from title by extracting the first word
        from Product title string
        Args:
            title_column: product title column
        Returns:
            split_text: first word of title (most likely brand but can be wrong as well)
        """
        split_text = title_column.split(" ")
        return split_text[0]

    @staticmethod
    def remove_commas(numeric_column: pd.Series) -> pd.Series:
        """
          This function fixes inconsistencies with the data. It removes ',' from numeric column and replaces with
          period.
          Args:
              numeric_column: numeric column from the dataframe
          Returns:
              numeric_column: column with consistent numeric format
        """
        numeric_column = numeric_column.replace(',', '', regex=True)
        return numeric_column

    @staticmethod
    def merge_ean_columns(product_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function aggregates all ean columns from master data, and further keeps only numeric distinct values
            Args:
                product_df: dataframe with ean columns that needs to be aggregated
            Returns:
                product_df: cleaned dataframe with merged ean column
        """
        product_df['merge_eans'] = product_df["p_eans"] + product_df["cp_eans"] + product_df["cp_un_eans"]
        product_df.drop(['p_eans', 'cp_eans', 'cp_un_eans'], axis=1, inplace=True)
        product_df['merge_eans'] = product_df['merge_eans'].apply(str)
        product_df['merge_eans'] = [list(dict.fromkeys([x for x in group if x.isdigit()])) for group in
                                    product_df['merge_eans']]
        return product_df

    @staticmethod
    def calculate_additional_features(clean_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function derives some additional attributes based on existing variables
        Args:
            clean_df: dataframe that needs to be further expanded using existing attributes and creating
            handcrafted features
        Returns:
            clean_df: extended dataframe with new features derived
        """
        # hybrid score using weighted combination of levenshtein and jaro wrinkler score
        clean_df['wg_sim_score'] = (0.7 * clean_df['lev']) + (0.3 * clean_df['jwrink'])
        # price outlier
        clean_df['price_diff'] = abs(clean_df['mas_dpc'] - clean_df['default_price_cents'])
        # EAN match - if supplier ean exists in our master data then 1 else 0
        clean_df['common_eans'] = [len(set(a).intersection(b)) for a, b in zip(clean_df.merge_eans, clean_df.eans)]
        clean_df['ean_match_flag'] = np.where(clean_df['common_eans'] > 0, 1, 0)
        if "size" in clean_df.columns:
            clean_df['size_diff'] = abs(clean_df['mas_size'] - clean_df['size'].astype(float))
        return clean_df

    def extract_features_from_title(self, product_df, is_tobacco_term) -> pd.DataFrame:
        """
        This function uses Title Miner module to extract additional values from
        supplier title column such as pieces, pieces count, size, unit
        Args:
            product_df: dataframe containing supplier title for products
            is_tobacco_term: if tobacco products TRUE else FALSE
        Returns:
            output_df: dataframe with additional extracted features
        """
        product_df['title_extract'] = product_df['title'].apply(extract_from_title.TitleMiner(
            product_df['title'], is_tobacco_term).get_pieces_size_unit)
        product_df[['term', 'extraction']] = product_df.title_extract.apply(pd.Series)
        # extract brand feature
        product_df['brand'] = product_df['title'].apply(self.extract_brand_from_product_title)
        # convert dict values to pandas columns eg. extracted pieces value is converted as column
        temp_df = product_df['extraction'].apply(pd.Series)
        output_df = pd.concat([product_df, temp_df], axis=1).reindex(product_df.index)
        output_df.drop(['term', 'title_extract', 'extraction'], axis=1, inplace=True)
        return output_df

    def fix_column_data_types(self, product_df: pd.DataFrame) -> pd.DataFrame:
        """ This function corrects data types of some specified columns, which are further used
         for mathematical calculation
         Args:
             product_df: dataframe consisting features to be used for further derivation
         Returns:
             output_df: dataframe with corrected data types for cols
        """
        output_df = product_df.replace(r'^\s*$', np.nan, regex=True)
        numeric_columns = ['mas_dpc', 'default_price_cents', 'mas_pieces', 'pieces', 'pieces_count', 'mas_p_c',
                           'mas_size', 'size']
        for column in numeric_columns:
            # sometimes these columns are absent as they're formed by extraction
            if column not in output_df.columns:
                pass
            else:
                # convert comma in size column to period
                output_df[column] = self.remove_commas(output_df[column])
                # convert to numeric types
                # output_df[column] = output_df[column].fillna(0)
                output_df[column] = output_df[column].fillna(0).astype(float)
        return output_df

    def execute(self, product_df, is_tobacco_flag=False):
        """ This function applies various data preprocessing functions defined above
         Args:
             product_df: product dataframe formed by blocking
             is_tobacco_flag: to specify if the tobacco flag is True/False
         Returns:
             output_df: preprocessed dataframe
        """
        output_df = self.process_ean_columns(product_df)
        # merge ean columns
        output_df = self.merge_ean_columns(output_df)
        # mine features from product title
        output_df = self.extract_features_from_title(output_df, is_tobacco_flag)
        # fix some data issues with column data types
        output_df = self.fix_column_data_types(output_df)
        # calculate handcrafted features
        output_df = self.calculate_additional_features(output_df)
        return output_df

