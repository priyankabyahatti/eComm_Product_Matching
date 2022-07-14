""" This script trains several different classification algorithms such as
 Decision Trees, Logistic Regression etc and fetches predictions """
import os
import pathlib
import pickle
import pandas as pd
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import DataPreprocessing
import settings
import glob


class MLClassification:
	"""The ml classification class is used to perform product matching using ml modelling"""
	@staticmethod
	def divide_train_test(x_data, y, test_size) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
		""" This function divides ML train data set into train, test for X and Y
		Args:
			x_data: ML Train data with labels
			y: target variable
			test_size: ratio of data to keep for testing
		Returns:
			X_train, X_test, y_train, y_test: All the split dataframes """
		X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size=test_size, random_state=42)
		return X_train, X_test, y_train, y_test

	@staticmethod
	def data_standardisation(x) -> pd.DataFrame:
		"""
		This function standardizes all independent variables so as to bring them to all one scale unit
		Args:
			x: dataframe with independent variables
		Returns:
			scaled_data: standardized dataframe for training
		"""
		scale = StandardScaler()
		scaled_data = scale.fit_transform(x)
		return scaled_data

	@staticmethod
	def get_feature_importance(model, model_name):
		"""
		This function prints importance of each variable in a particular ML model
		Args:
			model: any machine learning classification model
			model_name: name of ml model
		Returns: None
		"""
		# get importance
		if model_name == 'Logistic_Regression':
			importance = model.coef_[0]
		elif model_name == 'linear_SVM':
			importance = ''
		else:
			importance = model.feature_importances_
		# summarize feature importance
		for i, v in enumerate(importance):
			print(f'{model_name} Feature: %0d, Score: %.5f' % (i, v))
		pyplot.bar([x for x in range(len(importance))], importance)
		pyplot.show()

	@staticmethod
	def get_prediction(model, x_train, y_train, x_test):
		"""
		This function prints evaluation results of each model
		Args:
			model: any machine learning classification model
			x_train: train data with independent variables
			y_train: train data with dependent variables
			x_test: train data with independent variables
		Returns:
			y_pred: predicted target results
		"""
		model.fit(x_train, y_train)
		# Predict on test
		y_pred = model.predict(x_test)
		return y_pred

	def process_data_for_ml(self, train_df):
		# independent variables
		X = train_df.loc[:, train_df.columns != 'match'].values
		# dependent variable
		y = train_df['match'].values
		# standardize data
		X_scaled = self.data_standardisation(X)
		# split into train and test
		X_train, X_test, y_train, y_test = self.divide_train_test(X_scaled, y, 0.25)
		# Replace NaN with zero and infinity with large finite numbers
		X_train = np.nan_to_num(X_train)
		X_test = np.nan_to_num(X_test)
		return X_train, X_test, y_train, y_test

	@staticmethod
	def get_latest_training_data() -> pd.DataFrame:
		list_of_files = glob.glob(f"{settings.project_dir}" + "/training_data/*")
		latest_file = max(list_of_files, key=os.path.getmtime)
		train_df = pd.read_csv(latest_file)
		return train_df

	def execute(self):
		dp = DataPreprocessing()
		# fetch ml train data
		train_df = self.get_latest_training_data()
		print("labelled data generated. ")
		# feature selection
		train_df = train_df[['wg_sim_score',  'price_diff', 'ean_match_flag', 'size_diff', 'match']]
		# split into train and test
		X_train, X_test, y_train, y_test = self.process_data_for_ml(train_df)
		# Random Forests model
		rf_fit = RandomForestClassifier(n_estimators=100)
		# Logistic Regression
		lr_fit = LogisticRegression()
		# Decision Trees
		dt_fit = DecisionTreeClassifier()
		# XGBoost
		xb_fit = XGBClassifier()
		# models list
		model_list = [('Random_Forests', rf_fit), ('Logistic_Regression', lr_fit), ('Decision_Trees', dt_fit),
					  ('XG_Boost', xb_fit)]
		for model_name, model in model_list:
			print(" ***** "+model_name+" ***** ")
			y_pred = self.get_prediction(model, X_train, y_train, X_test)
			dp.calculate_performance_evaluation(y_test, y_pred, "ML-based Results")
			self.get_feature_importance(model, model_name)
			# save models locally
			filename = settings.project_dir+"ml_models/"+model_name
			pickle.dump(model, open(filename+".pkl", "wb"))


if __name__ == '__main__':
	MLClassification().execute()
	print("Generated ML-Based Matched Results. ")
