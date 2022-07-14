""" Blocking is a technique that helps reducing the similarity search space. This blocking script
 produces potential candidate matches for every product title using either TF-IDF based blocking, or
 Standard blocking using block key, or both. Thereby reduces O(n2) complexity."""
import numpy as np
import pandas as pd
from recordlinkage.index import Block
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class HybridBlocking:
    """This class can be used to implement either standard blocking, tf-idf based blocking or both"""
    def __init__(self, source_df, target_df, source_title, target_title, std_src_attr=None, std_target_attr=None):
        """The class initialized with following parameters
        Args:
            source_df: df containing supplier inventory data
            target_df: df containing master data
            source_title: product title from supplier source
            target_title: product title from master data
            std_src_attr: block key in this case category id from source
            std_target_attr: block key in this case category id from target
        """
        self.df_1 = source_df
        self.df_2 = target_df
        self.source_title = source_title
        self.target_title = target_title
        self.std_src_attribute = std_src_attr
        self.std_target_attribute = std_target_attr

    @staticmethod
    def generate_ngrams(string, n=4) -> list[str]:
        """
        This functions generates ngrams for sentences for eg: Marlboro Rot -> 'Marl', 'arlb', 'boro' etc
            Args:
                string: the product title to break into ngrams
                n: the size of ngrams
            Returns:
                tokens: returns title into groups of ngram tokens
        """
        # Convert to lower-cases
        s = string.lower()
        # Replace all none alphanumeric characters with spaces
        s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
        s = re.sub(r'[,-./]|\sBD', r'', s)
        # Break sentence in the token, remove empty tokens
        # Use the zip function to help us generate n-grams
        ngrams = zip(*[s[i:] for i in range(n)])
        tokens = ["".join(ngram) for ngram in ngrams]
        return tokens

    @staticmethod
    def drop_na_df(df_1, df_2):
        """
        This functions removes nan values from title columns of both source and target dataframe
            Args:
                df_1: dataframe containing inventory product title
                df_2: dataframe containing master data product title
            Returns:
                df_1, df_2: returns both dataframes without nan values in titles
        """
        df_1 = df_1[df_1['title'].notna()]
        df_2 = df_2[df_2['mas_title'].notna()]
        return df_1, df_2

    @staticmethod
    def get_top_n_candidates(row, top_n=100):
        """
        This functions generates top n most similar candidates for comparision
            Args:
                row: values from the dot product
                top_n: the number of potential matches to be selected
            Returns:
                tokens: sorted top n values
        """
        row_count = row.getnnz()
        if row_count == 0:
            return None
        elif row_count <= top_n:
            result = zip(row.indices, row.data)
        else:
            arg_idx = np.argpartition(row.data, -top_n)[-top_n:]
            result = zip(row.indices[arg_idx], row.data[arg_idx])
        return sorted(result, key=(lambda x: -x[1]))

    def match_product(self, supplier_title, vectorizer, master_product_vectors, supp_index, result_list):
        """
        This functions generates top n most similar candidates for comparison
            Args:
                supplier_title: supplier title to be compared against
                vectorizer: initialized tfidf vectorizer with defined ngrams
                master_product_vectors: title vectors from th whole master data
                supp_index: supplier product index
                result_list: total top n list pf products
            Returns:
                topn_list: a list with top N matching titles with match score
        """
        input_name_vector = vectorizer.transform([supplier_title])
        result_vector = input_name_vector.dot(master_product_vectors.T)
        # top n results
        matched_data = [self.get_top_n_candidates(row) for row in result_vector]
        if None in matched_data:
            return None
        else:
            flat_matched_data = [tup[0] for data_row in matched_data for tup in data_row]
            supp_idx = [supp_index] * len(flat_matched_data)
            topn_list = list(zip(supp_idx, flat_matched_data))
            result_list.append(topn_list)
            return topn_list

    def tfidf_based_blocking(self) -> pd.MultiIndex:
        """
        This functions executes tfidf based blocking
            Returns:
                results: multi-index list of tuples containing pair of two product indices
        """
        df_1, df_2 = self.drop_na_df(self.df_1, self.df_2)
        vectorizer = TfidfVectorizer(min_df=1, analyzer=self.generate_ngrams, lowercase=True)
        product_vectors = vectorizer.fit_transform(df_2[self.target_title])
        results = []
        df_1.apply(lambda row: self.match_product(row[self.source_title], vectorizer, product_vectors, row.name,
                                                  results), axis=1)
        flatten_results = [val for sublist in results for val in sublist]
        results = pd.MultiIndex.from_tuples(flatten_results)
        return results

    def standard_category_blocking(self) -> pd.MultiIndex:
        """
        This functions executes standard blocking on category column
            Returns:
                results: multi-index list of tuples containing pair of two product indices
        """
        block_idx_category = Block(left_on=self.std_src_attribute, right_on=self.std_target_attribute)
        results = block_idx_category.index(self.df_1, self.df_2)
        return results

    def execute(self):
        """
        This functions executes standard blocking and tfidf based blocking
            Returns:
                results: multi-index list of tuples containing pair of two product indices
        """
        # std blocking
        category_based_candidates = self.standard_category_blocking()
        # tfidf blocking
        tfidf_based_candidates = self.tfidf_based_blocking()
        candidate_pairs = category_based_candidates.append(tfidf_based_candidates)
        results = candidate_pairs.drop_duplicates(keep="first")
        return results

