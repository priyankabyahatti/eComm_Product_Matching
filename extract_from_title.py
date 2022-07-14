"""
this module holds all the functions to extract data from
the product title
"""
import re
from typing import Tuple, List
import pandas as pd


class TitleMiner():
    """
    holds the logic to mine information from the inventory item title
    :method
    """

    def __init__(self,
                 title: str = '',
                 is_tobacco_term: bool = False):
        """
        initializes the class
        """
        self.title: str = title
        self.is_tobacco_term = is_tobacco_term

    @staticmethod
    def __standardize_size(size: str) -> float:

        """
        for the size we do expect a standard float
        in python. however as we import excels from germans
        those sometimes contains also , as decimal separator
        :param size: size as floatalike string
        :return:
        """
        size = str(size).replace(',', '.')
        return float(size)

    def __recalculate_units(self, input_unit: str, indicator: List[str], output_unit: str,
                            size: float, factor: float) -> Tuple[float, str]:
        if input_unit in indicator:
            unit = output_unit.lower()
            size = size * factor
            return size, unit
        return size, input_unit

    def __standardize_units(self, size: float, unit: str) -> Tuple[float, str]:
        """
            we do expect certain keywords within the database
            therefore we standardize to unit signs like g & l
            also it is save to expect a litre unit if no unit was given
            and the size matches the common sizes of bottles etc
            we also standardize ml to l and kg to g. in these cases
            the sizes get adjusted

            :param size: the size of the product
            :param unit: the current unit of the product
            :return: a standardized unit of the product
         """
        if unit is None:
            unit = ''
        unit = unit.lower()
        unit = re.sub(r"gramm|gram|grm|gr", "g", unit)
        unit = re.sub(r"litre|liter|ltr|lt|i", "l", unit)
        common_litres = [0.02, 0.05, 0.04, 0.1, 0.2, 0.25, 0.3, 0.33, 0.4,
                         0.5, 0.70, 0.75, 1.00, 1.5]
        size, unit = self.__recalculate_units(unit, ["ml"], "l", size, 0.001)
        size, unit = self.__recalculate_units(unit, ["cl"], "l", size, 0.01)
        size, unit = self.__recalculate_units(unit, ["kg", "kilo"], "g", size, 1000)
        size, unit = self.__recalculate_units(unit, ["m"], "cm", size, 100)
        size, unit = self.__recalculate_units(unit, ["mm"], "cm", size, 0.1)

        if (size in common_litres) & (unit == ''):
            unit = 'l'

        return size, unit

    @staticmethod
    def __order_regex_results(term: str,
                              regex_options):
        """
        loops through all regex_options and updates the dict accordingly
        
        :param term: the term which nees to be analyzed and modified
        :param regex_options: the to be analyzed regex strings as list
        :return: results as dict.
        sample:  {'term': term,
                  'extraction': {}}
        """
        result = {'term': term,
                  'extraction': {}}
        for regex_option in regex_options:
            pattern_obj = re.compile(regex_option, flags=re.I)
            if pattern_obj.search(term):
                dict_to_rename = pattern_obj.search(term).groupdict()
                result['extraction'].update(dict_to_rename)
                result['term'] = pattern_obj.sub(repl='', string=term)
                term = result['term']

        return result

    @staticmethod
    def __standardize_title(term: str) -> str:
        """
        bring term to the wanted title format 
        (stripped, title & remove duplicate whitespaces)
        
        :param term: the term which nees to be analyzed and modified
        :return: cleaned term
        """
        term = term.strip()
        term = re.sub(r"\s+", " ", term)
        term = term.title()

        return term

    @staticmethod
    def __extract_and_replace_regex(term: str,
                                    regex_options) -> dict:
        """
        extract regex from string, replace with empty string
        
        :param term: the term which nees to be analyzed and modified 
        :param regex_options: the to be analyzed regex strings as list
        :return: result as dict. 
        sample:  {'term': term,
                  'extraction': {}}
        """
        result = {'term': term,
                  'extraction': {}}
        if isinstance(regex_options, str):
            regex_options = [regex_options]
        for regex_option in regex_options:
            pattern_obj = re.compile(regex_option, flags=re.I)
            if pattern_obj.search(term):
                result['extraction'] = pattern_obj.search(term).groupdict()
                result['term'] = pattern_obj.sub(repl='', string=term)
                break

        return result

    def get_product_format(self,
                           term: str) -> dict:
        """
        looks for format information within the given term 
        
        :param term: the term which nees to be analyzed and modified
        :return: dict of results 
        """
        regex = r"\b(?P<format>DS|Dose|Dos|Pet|Glas)\b"
        extracted_data = self.__extract_and_replace_regex(term=term,
                                                          regex_options=regex)

        return extracted_data

    def get_alcohol_percentage(self,
                               term: str) -> dict:
        """
            gets alcohol percentage information about the produc
            
            :param term: the term which nees to be analyzed and modified
            :return:  dict of results 
            """
        regex = r"(?P<alcohol>\d+.?\d*\s*\%)"
        extracted_data = self.__extract_and_replace_regex(term=term,
                                                          regex_options=regex)

        return extracted_data

    def get_pieces_from_pack_title(self,
                                   term: str) -> dict:
        """
        when a pack title is extracted we need to generate the pieces out of it
        Args:
            term (): the term of the pack_title

        Returns: the amount of pieces
        """
        regex = r"(?P<pieces_1>\d+)\s*[\*xà\/]\s*(?P<pieces_2>\d+)"
        extraction = self.__extract_and_replace_regex(term=term,
                                                      regex_options=regex)['extraction']
        pieces = int(extraction.get('pieces_1', 0)) * int(extraction.get('pieces_2', 0))

        return pieces

    def get_pieces_size_unit(self,
                             term: str) -> dict:
        """
        gets pieces, size, unit & pieces count information about the product
        
        :param term: the term which nees to be analyzed and modified
        :return:  dict of results 
        """
        # define regexp elements
        # pylint: disable=line-too-long, too-many-locals
        regex_pieces = r'(?P<pieces>\d+)'
        regex_pieces_count = r'(?P<pieces_count>\d+)'
        regex_delimiter = r'[\*xà]'
        regex_size = r"(?P<size>\d+(?:[.,]?\d*))"
        regex_size_decimal_mandatory = r"(?P<size>\d+(?:[.,]\d*))"
        regex_size_fallback = r"(?P<size>\d+[.,]\d+)"
        regex_unit = r"(?P<unit>liter|litre|ltr|gramm|gr|ml|cl|kg|g|l|Blatt|LT|cm|€|I\b)"
        regex_stück = r"(?P<pieces_count>\d+)\s*(?:Stück|Stck|Stuck|Stueck|Stk|St|Dragees|Drages|Rollen|Dr|Riegel|Stangen|Kaugummis\.?)\b"
        regex_stück_pieces = r"(?P<pieces>\d+)\s*(?:Stück|Stck|Stuck|Stueck|Stk|St\.?)\b"
        regex_beutel = r"(?P<pieces>\d+)\s*(?:Beutel|Btl\.?)"
        regex_er = r"(?P<pieces_count>\d+)\s*[´]*(?:er|Pack\.?)\b"
        regex_pack_title = r"(?P<pack_title>\d+\s*[\*xà\/]\s*\d+)"
        regex_pieces_delimiter_pieces_count = rf"{regex_pieces}\s*{regex_delimiter}\s*{regex_pieces_count}"
        regex_pack_title_size_unit = rf"{regex_pack_title}\s*{regex_delimiter}\s*{regex_size}\s*{regex_unit}"
        regex_pack_title_size = rf"{regex_pack_title}\s*{regex_size_decimal_mandatory}"
        regex_pieces_pieces_count_slash_size_unit = rf"{regex_pieces}\s*{regex_delimiter}\s*{regex_pieces_count}\s*\/\s*{regex_size}\s*{regex_unit}?.*\b"
        regex_er_pieces_size_unit = rf"{regex_er}\s*{regex_pieces}\s*{regex_delimiter}\s*{regex_size}\s*{regex_unit}?.*\b"
        regex_pieces_size_unit_mandatory = rf"{regex_pieces}\s*{regex_delimiter}\s*{regex_size}\s*{regex_unit}\b"
        regex_pieces_size_unit = rf"{regex_pieces}\s*{regex_delimiter}\s*{regex_size}\s*{regex_unit}?.*\b"
        regex_pieces_pieces_count_description_size_unit = rf"{regex_pieces}\s*{regex_delimiter}[a-zA-Z\s,]*{regex_pieces_count}[a-zA-Z\s]+{regex_size}\s*{regex_unit}?.*\b"
        regex_pieces_count_pieces_size_unit = rf"{regex_pieces}\s*{regex_delimiter}[\s\(]*{regex_pieces_count}\s*{regex_delimiter}\s*{regex_size}\s*{regex_unit}?.*\b"
        regex_pieces_stück = rf"{regex_pieces}\s*{regex_delimiter}\s*{regex_stück}"
        regex_size_slash_pieces = rf"{regex_size}\/{regex_pieces}"
        regex_size_unit_stück = rf"{regex_size}\s*{regex_unit}\s*{regex_stück}"
        regex_stück_slash_size = rf"{regex_stück}\/{regex_size}\s*{regex_unit}"
        regex_stück_delimiter_size = rf"{regex_stück_pieces}\s*[xà]\s*{regex_size}\s*{regex_unit}"
        regex_stück_size = rf"{regex_stück_pieces}\s*{regex_size}\s*{regex_unit}"
        regex_beutel_size = rf"{regex_beutel}\s*{regex_size}\s*{regex_unit}"
        regex_size_unit_pieces = rf"{regex_size}\s*{regex_unit}\s*{regex_delimiter}\s*{regex_pieces}"
        regex_size_unit = rf"{regex_size}\s*{regex_unit}\.*\b"
        regex_pieces_count_delimiter = rf"{regex_pieces_count}\s*{regex_delimiter}"
        regex_pieces_er = rf"{regex_pieces}\s*{regex_delimiter}\s*{regex_er}"
        regex_size_unit_pieces_er = rf"{regex_size}\s*{regex_unit}\s*{regex_er}"
        regex_delimiter_pieces = rf"{regex_delimiter}\s*{regex_pieces}"
        regex_pieces_size_unit_er = rf"{regex_pieces}\s*{regex_delimiter}\s*{regex_size}\s*{regex_unit}\s*{regex_er}"
        regex_tobacco_pieces_count = rf"{regex_pieces_count}"

        # execute for all qualified regexp compositions
        regex_options = [regex_pack_title_size_unit,
                         regex_pieces_pieces_count_slash_size_unit,
                         regex_stück_slash_size,
                         regex_pieces_size_unit_er,
                         regex_stück_delimiter_size,
                         regex_size_unit_stück,
                         regex_size_unit_pieces_er,
                         regex_stück_size,
                         regex_pieces_stück,
                         regex_stück,
                         regex_pack_title_size,
                         regex_beutel_size,
                         regex_pieces_count_pieces_size_unit,
                         regex_er_pieces_size_unit,
                         regex_pieces_pieces_count_description_size_unit,
                         regex_pieces_er,
                         regex_pieces_size_unit_mandatory,
                         regex_pieces_size_unit,
                         regex_size_unit_pieces,
                         regex_size_unit,
                         regex_er,
                         regex_size_slash_pieces,
                         regex_pieces_count_delimiter,
                         regex_delimiter_pieces,
                         regex_size_fallback]

        if self.is_tobacco_term:
            # another interpration of integer times integer
            # needs to be inserted before regex_pieces_size_unit
            # in order to catch things like Marlboro Red 10x20
            regex_options.insert(15, regex_pieces_delimiter_pieces_count)
            regex_options.append(regex_tobacco_pieces_count)

        extracted_data = self.__order_regex_results(term=term,
                                                    regex_options=regex_options)

        return extracted_data

    def get_packaging(self, term: str) -> dict:
        """
        gets packaging information about the product

        :param term: the term which nees to be analyzed and modified
        :return: dict of results 
        """
        regex = r'\b(?P<packaging>Kar?ton|Kiste)\b'
        extracted_data = self.__extract_and_replace_regex(term=term,
                                                          regex_options=regex)

        return extracted_data

    def get_dpg(self, term: str) -> dict:
        """
        gets deposit information about the product

        :param term: the term which nees to be analyzed and modified
        :return: dict of results
        """
        regex = r'\b(?P<dpg>DPG|MW|EW)\b'
        extracted_data = self.__extract_and_replace_regex(term=term,
                                                          regex_options=regex)

        return extracted_data

    def get_kvp(self, term: str) -> dict:
        """
        gets kvp information about the product
        Args:
            term: the term which nees to be analyzed and modified

        Returns: dict of results

        """
        regex = r"(?P<kvp>\d+[.,]\d+)\s*(?P<kvp_unit>€)?"
        extracted_data = self.__extract_and_replace_regex(term=term,
                                                          regex_options=regex)

        return extracted_data

    def get_pack_title_from_context(self,
                                    full_result_dict: dict):
        """
        for some parts we do have pretty complex patterns which indicates on a pack title
        without the chance of directly catch all of those
        Returns: the updated, if it was necessary, full result dict

        """
        common_beverages_sizes = [0.2, 0.25, 0.3, 0.33, 0.4, 0.5]
        if full_result_dict['unit'].lower() in ['l'] \
                and float(full_result_dict['size']) in common_beverages_sizes:
            pieces = int(full_result_dict['pieces_count']) * int(full_result_dict['pieces'])
            pack_title = f"{full_result_dict['pieces']}x{full_result_dict['pieces_count']}"
            full_result_dict['pack_title'] = pack_title
            full_result_dict['pieces'] = pieces
            full_result_dict.pop('pieces_count')

        return full_result_dict

    def normalize_pack_title(self,
                             term):
        """
        replaces the delimiter of a term to our normalized pack title delimiter "x"
        Args:
            term (): the given term to normalize

        Returns: the normalized pack title

        """
        return re.sub(r'[\*xà]', 'x', term)

    def execute(self):
        """
        executes the whole extraction pipeline and consolidates the results

        :return: dict of the result f.e.
                                          {'alcohol': '40 %',
                                           'dpg': 'DPG',
                                           'packaging': 'Kiste',
                                           'pieces': '4',
                                           'pieces_count': '6',
                                           'size': 0.7,
                                           'title': 'Jack Daniels Whiskey',
                                           'unit': 'l'}
        """
        full_result_dict = {}
        title = self.title
        if self.is_tobacco_term:
            kvp_dict = self.get_kvp(title)
            full_result_dict.update(kvp_dict['extraction'])
            title = kvp_dict['term']
        dpg_dict = self.get_dpg(title)
        title = dpg_dict['term']
        full_result_dict.update(dpg_dict['extraction'])
        alcohol_dict = self.get_alcohol_percentage(title)
        full_result_dict.update(alcohol_dict['extraction'])
        title = alcohol_dict['term']
        packaging_dict = self.get_packaging(title)
        full_result_dict.update(packaging_dict['extraction'])
        title = packaging_dict['term']
        format_dict = self.get_product_format(title)
        full_result_dict.update(format_dict['extraction'])
        title = format_dict['term']
        size_unit_pieces_count_dict = self.get_pieces_size_unit(title)
        full_result_dict.update(size_unit_pieces_count_dict['extraction'])
        title = size_unit_pieces_count_dict['term']
        full_result_dict['title'] = self.__standardize_title(title)
        if 'size' in full_result_dict.keys():
            full_result_dict['size'] = self.__standardize_size(full_result_dict['size'])
            if 'unit' in full_result_dict.keys():
                full_result_dict['size'], full_result_dict[
                    'unit'] = self.__standardize_units(
                    full_result_dict['size'],
                    full_result_dict['unit'])
            if ('pack_title' not in full_result_dict.keys()) and all(
                    column in full_result_dict.keys() for column in ['pieces_count', 'unit', 'pieces']):
                full_result_dict = self.get_pack_title_from_context(full_result_dict)
            if 'pack_title' in full_result_dict.keys():
                full_result_dict['pieces'] = self.get_pieces_from_pack_title(full_result_dict['pack_title'])
                full_result_dict['pack_title'] = self.normalize_pack_title(full_result_dict['pack_title'])

        return full_result_dict


class TitleMinerDataFrameExecutor:
    """
    this class holds the logic to execute the title miner over a dataframe
    """

    def __init__(self,
                 dataframe_to_work_on: pd.DataFrame,
                 is_tobacco_list: bool = True):
        """
        initializes the class

        :param dataframe_to_work_on: dataframe where the title miner should be executed against
        """
        self.dataframe = dataframe_to_work_on
        self.original_column_prefix: str = 'original_'
        self.title_column_name: str = 'title'
        self.is_tobacco_list: bool = is_tobacco_list

    @staticmethod
    def __check_dataframe_column_presence(dataframe_to_check: pd.DataFrame) -> bool:
        """
        :param dataframe_to_check: pandas dataframe to check the needed column presence
        :return: boolean if the needed columns are there
        """
        columns_to_be_present = ['title']
        for column_to_be_present in columns_to_be_present:
            if column_to_be_present in dataframe_to_check.columns:
                return True
            else:
                raise ValueError(
                    f"expected column: {column_to_be_present} is not presence in dataframe")

    @staticmethod
    def __apply_title_miner_to_rows(dataframe_to_be_worked_on: pd.DataFrame,
                                    title_column_name: str,
                                    is_tobacco_list: bool = False) -> pd.DataFrame:
        """
        runs the title miner per column and adds those information to the dataframe
        :param dataframe_to_be_worked_on: the to be worked on dataframe
        :param title_column_name: the column name of the title column
        :return: dataframe with mined informations as new columns
        """
        dataframe_to_be_worked_on.reset_index(inplace=True)
        dataframe_to_store_results = pd.DataFrame()
        for k, row in dataframe_to_be_worked_on.iterrows():
            resultset = TitleMiner(row[title_column_name],
                                   is_tobacco_term=is_tobacco_list).execute()
            dataframe_to_store_results = dataframe_to_store_results.append(resultset,
                                                                           ignore_index=True)
        dataframe_with_extended_information = dataframe_to_be_worked_on.join(
            dataframe_to_store_results, lsuffix='_original')

        return dataframe_with_extended_information

    def execute(self):
        """
        executes the title miner logic over a given dataframe

        :return: the new dataframe with the extended information as new columns
        """
        if self.__check_dataframe_column_presence(self.dataframe):
            dataframe_with_extended_information = self.__apply_title_miner_to_rows(
                dataframe_to_be_worked_on=self.dataframe,
                title_column_name=self.title_column_name,
                is_tobacco_list=self.is_tobacco_list)

            return dataframe_with_extended_information
