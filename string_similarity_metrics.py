""" This script is used in calculating several distance measures such as edit based and token based
metrics for any given two strings """

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
import pandas as pd
from fuzzywuzzy import fuzz
import py_stringmatching as sm
import textdistance as td


class StringMetrics:
    """
       holds the logic to calculate string similarity score between two strings
       """
    def __init__(self, input_df=None):
        """
            initializes the class with optional parameter
            Args:
                input_df: the dataframe which consists two columns with product titles
        """
        self.input_df = input_df

    @staticmethod
    def __convert_text_to_tokens(str_1, str_2) -> tuple[list, list]:
        """
        Converts text value into group of tokens which is required input for token based metrics
        Args:
            str_1: string 1 with product title
            str_2: string 2 with product title
        Returns:
            token_list1, token_list2: token lists containing group of split tokens
        """
        token_list1 = str_1.split()
        token_list2 = str_2.split()
        return token_list1, token_list2

    @staticmethod
    def initialize_str_metric_objs() -> tuple[list, list]:
        """
        Initializes all string metric objects from py_stringmatching library
        Returns:
            edit_obj_list, token_obj_list: lists containing edit and token metric objects
        """
        jaro = sm.Jaro()
        jwrink = sm.JaroWinkler()
        cos = sm.Cosine()
        stfidf = sm.SoftTfIdf()
        jacrd = sm.Jaccard()
        melkan = sm.MongeElkan()
        hamm = td.Hamming()
        ratcliff = td.RatcliffObershelp()
        # lists
        edit_obj_list = [jaro, jwrink, hamm, ratcliff]
        token_obj_list = [cos, stfidf, jacrd, melkan]
        return edit_obj_list, token_obj_list


    @classmethod
    def get_edit_metric_scores(cls, str_1, str_2) -> list:
        """
        Gets edit based metric scores
        Args:
            str_1: string 1 with product title
            str_2: string 2 with product title
        Returns:
            [jaro, jwrink, lev, hamm, dlev]: list of edit based scores of two strings
        """
        jaro, jwrink, hamm, ratcliff = cls.initialize_str_metric_objs()[0]
        # calculate edit scores
        jaro = jaro.get_sim_score(str_1, str_2)
        jwrink = jwrink.get_sim_score(str_1, str_2)
        lev = fuzz.token_set_ratio(str_1, str_2) / 100
        hamm = hamm.normalized_similarity(str_1, str_2)
        ratcliff = ratcliff.normalized_similarity(str_1, str_2)
        dlev = normalized_damerau_levenshtein_distance(str_1, str_2)
        return [jaro, jwrink, lev, hamm, dlev, ratcliff]

    @classmethod
    def get_token_metric_scores(cls, str_1, str_2) -> list:
        """
        Gets token based metric scores
        Args:
            str_1: string 1 with product title
            str_2: string 2 with product title
        Returns:
            [melkan, jacrd, cos, stfidf]: list of token based scores of two strings
        """
        token_list1, token_list2 = cls.__convert_text_to_tokens(str_1, str_2)
        cos, stfidf, jacrd, melkan = cls.initialize_str_metric_objs()[1]
        # calculate token based scores
        jacrd = jacrd.get_sim_score(token_list1, token_list2)
        cos = cos.get_sim_score(token_list1, token_list2)
        stfidf = stfidf.get_raw_score(token_list1, token_list2)
        melkan = melkan.get_raw_score(token_list1, token_list2)
        return [melkan, jacrd, cos, stfidf]

    def get_all_sim_scores(self, str_1, str_2) -> list:
        """
        This function gets all similarity scores for any given two strings
        Args:
            str_1: string 1 with product title
            str_2: string 2 with product title
        Returns:
            distance_metric_list: list of all scores for any two strings
        """
        # edit based
        jaro, jwrink, lev, hamm, dlev, ratcliff = self.get_edit_metric_scores(str_1, str_2)
        # token based
        cos, stfidf, jacrd, melkan = self.get_token_metric_scores(str_1, str_2)
        distance_metric_list = [jaro, jwrink, lev, dlev, ratcliff, hamm, cos, jacrd, melkan, stfidf]
        return distance_metric_list

    def execute(self) -> pd.DataFrame:
        """
        This function gets all similarity scores for any given two strings present in a dataframe, and creates
        individual distance score columns in the same dataframe

        Returns:
            output_df: output dataframe that contains all distance scores as new columns
        """
        self.input_df[['scores']] = \
            self.input_df.apply(lambda x: self.get_all_sim_scores(x['title'], x['mas_title']), axis=1)
        self.input_df[['jaro', 'jwrink', 'lev', 'dlev', 'ratcliff', 'hamm', 'cos', 'jacrd', 'melkan', 'stfidf']] \
            = pd.DataFrame(self.input_df.scores.tolist(), index=self.input_df.index)
        output_df = self.input_df.drop('scores', axis=1)
        return output_df



