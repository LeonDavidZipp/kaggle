import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np
import sklearn.decomposition as sd
import sklearn.pipeline as sp
import sklearn.model_selection as sm
import sklearn.preprocessing as spp
import sklearn.compose as sc
import sklearn.impute as si
import sklearn.feature_extraction as sf
import statsmodels as sm
import feature_engine.creation as fc


class SpaceTitanicTransformer:
	def __init__(self):
		target = [
			"Transported"
		]
		cat_cols = [
			"Group", "CryoSleep", "Destination", "Age", "VIP", "CabinDeck", "CabinNum", "CabinSide"
		]
		self.col_tf = sc.make_column_transformer(
			# (spp.OneHotEncoder(handle_unknown="ignore"), cat_cols),
			(spp.OneHotEncoder(handle_unknown="ignore"), cat_cols),
			("passthrough", target)
		)
		self.added_y_col = False

	def transform(self, X:pl.LazyFrame, y=None):
		X = self._transform(X)
		data = self.col_tf.transform(X)
		if self.added_y_col:
			return data[:, :-1]
		else:
			return data

	def fit(self, X: pl.LazyFrame, y=None):
		X = self._transform(X)
		self.col_tf.fit(X)

	def fit_transform(self, X: pl.LazyFrame, y=None):
		X = self._transform(X)
		data = self.col_tf.fit_transform(X)
		if self.added_y_col:
			return data[:, :-1]
		else:
			return data

	def _transform(self, X: pl.LazyFrame) -> pd.DataFrame:
		X = self.drop_cols(X)
		X = self.add_y_col_if_not_there(X)
		X = self.recast_and_handle_missings_when_appropriate(X)
		X = self.bin_age_col(X)
		X = self.extract_group_from_id_col(X)
		return self.split_cabin_col(X).collect().to_pandas()

	def drop_cols(self, lf: pl.LazyFrame):
		return (
			lf.drop(
				[
					"HomePlanet",
					# "Destination",
					"RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name"
				]
			)
		)

	def add_y_col_if_not_there(self, lf: pl.LazyFrame):
		df = lf.collect()
		if not "Transported" in df.columns:
			self.added_y_col = True
			return lf.with_columns(pl.Series(values=[None] * df.height).alias("Transported"))
		else:
			return lf

	def recast_and_handle_missings_when_appropriate(self, lf: pl.LazyFrame):
		return (
			lf.with_columns(
				pl.col("CryoSleep").cast(pl.Int8).cast(pl.String).fill_null(pl.lit("unknown")),
				pl.col("Cabin").fill_null(pl.lit("unknown/unknown/unknown")),
				pl.col("Destination").fill_null(pl.lit("unknown")),
				pl.col("VIP").cast(pl.Int8).cast(pl.String).fill_null(pl.lit("unknown")),
				pl.col("Transported").cast(pl.Int8)
			)
		)

	def bin_age_col(self, lf: pl.LazyFrame):
		breaks = [12, 18, 30, 50, 65]
		labels = ["Child", "Teenager", "Young Adult", "Adult", "Middle-aged", "Senior"]
		return (
			lf.with_columns(
				pl.col("Age").cut(breaks=breaks, labels=labels).fill_null(pl.lit("unknown")).cast(pl.String)
			)
		)

	def extract_group_from_id_col(self, lf: pl.LazyFrame):
		# into group
		return (
			lf.with_columns(
				pl.col("PassengerId").str.split(by="_").list.get(0).alias("Group")
			).drop(["PassengerId"])
		)

	def split_cabin_col(self, lf: pl.LazyFrame):
		# into deck, num, side
		return (
			lf.with_columns(
				pl.col("Cabin").str.split(by="/").list.get(0).alias("CabinDeck"),
				pl.col("Cabin").str.split(by="/").list.get(1).alias("CabinNum"),
				pl.col("Cabin").str.split(by="/").list.get(2).alias("CabinSide")
			).drop(["Cabin"])
		)
