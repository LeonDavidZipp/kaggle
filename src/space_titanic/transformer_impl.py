import polars as pl
import polars.selectors as cs
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
		# cat_cols = [
		# 	"Group", "CryoSleep", "Destination", "Age", "VIP", "CabinDeck", "CabinNum", "CabinSide"
		# ]
		self.col_tf = sc.make_column_transformer([
			(sf.FeatureHasher(n_features=9280), ["Group"]),
			(sf.FeatureHasher(n_features=3), ["CryoSleep"]),
			(sf.FeatureHasher(n_features=4), ["Destination"]),
			(sf.FeatureHasher(n_features=7), ["Age"]),
			(sf.FeatureHasher(n_features=3), ["VIP"]),
			(sf.FeatureHasher(n_features=6), ["CabinDeck"]), # determine beforehand, only placeholders now
			(sf.FeatureHasher(n_features=200), ["CabinNum"]), # determine beforehand, only placeholders now
			(sf.FeatureHasher(n_features=3), ["CabinSide"])
			("passthrough", target)
		])

	def transform(self, X:pl.LazyFrame, y=None):
		X = self.drop_cols(X)
		X = self.recast_and_handle_missings_when_appropriate(X)
		X = self.bin_age_col(X)
		X = self.extract_group_from_id_col
		X = self.split_cabin_col(X).collect().to_pandas()
		return self.col_tf.transform(X)

	def fit(self, X: pl.LazyFrame, y=None):
		X = self.drop_cols(X)
		X = self.recast_and_handle_missings_when_appropriate(X)
		X = self.bin_age_col(X)
		X = self.extract_group_from_id_col
		X = self.split_cabin_col(X).collect().to_pandas()
		self.col_tf.fit(X.collect().to_pandas)

	def fit_transform(self, X: pl.LazyFrame, y=None):
		self.fit(X)
		return self.col_tf.transform(X.collect().to_pandas())

	def drop_cols(lf: pl.LazyFrame):
		return (
			lf.drop(
				[
					"HomePlanet",
					# "Destination",
					"RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name"
				]
			)
		)

	def recast_and_handle_missings_when_appropriate(self, lf: pl.LazyFrame):
		return (
			lf.with_columns(
				pl.col("CryoSleep").cast(pl.Int8).cast(pl.String).fill_null(pl.lit("unknown")),
				pl.col("Cabin").fill_null(pl.lit("unknown/unknown/unknown")),
				pl.col("Destination").fill_null(pl.lit("unknown")),
				# pl.col("Age").cast(pl.Int8).cast(pl.String).fill_null(pl.lit("unknown")),
				pl.col("VIP").cast(pl.Int8).cast(pl.String).fill_null(pl.lit("unknown")),
			)
		)

	def bin_age_col(self, lf: pl.LazyFrame):
		bins = [0, 18, 30, 50, 65, 100]
		labels = ["Child", "Young Adult", "Adult", "Middle-aged", "Senior"]
		return (
			# lf.with_columns(
			# 	pl.when(pl.col("age").is_null()).then(pl.lit("Unknown"))
			# 	.when(pl.col("age") < 30).then(pl.lit("Young Adult"))
			# 	.when(pl.col("age") < 50).then(pl.lit("Adult"))
			# 	.when(pl.col("age") < 65).then(pl.lit("Middle-aged"))
			# 	.otherwise(pl.lit("Senior"))
			# )
			lf.with_columns(
				pl.col("Age").cut(breaks=bins, labels=labels).fill_null(pl.lit("unknown"))
			)
		)

	def extract_group_from_id_col(self, lf: pl.LazyFrame):
		# into group
		return (
			lf.with_columns(
				pl.col("PassengerId").str.split(by="_").get(0).alias("Group")
			).drop(["PassengerId"])
		)

	def split_cabin_col(self, lf: pl.LazyFrame):
		# into deck, num, side
		return (
			lf.with_columns(
				pl.col("Cabin").str.split(by="/").get(0).alias("CabinDeck"),
				pl.col("Cabin").str.split(by="/").get(1).alias("CabinNum"),
				pl.col("Cabin").str.split(by="/").get(2).alias("CabinSide")
			).drop(["Cabin"])
		)
