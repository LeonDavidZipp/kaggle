import polars as pl
import sklearn.base as sb
import sklearn.decomposition as sd
import sklearn.pipeline as sp
import sklearn.model_selection as sm
import sklearn.preprocessing as spp
import sklearn.compose as sc
import sklearn.impute as si


class RainfallTransformer(sb.BaseEstimator):
	def __init__(self):
		drop_cols = ["day", "dewpoint"]
		temp_cols = ["temparature", "mintemp", "maxtemp"]
		# for temperature, min- & maxtemp
		temp_pl = sp.make_pipeline(
			si.SimpleImputer(strategy="median"),
			spp.MinMaxScaler(),
			sd.PCA(n_components=1)
		)
		step_1_pl = sp.make_pipeline(
			si.SimpleImputer(strategy="median"),
			spp.MinMaxScaler(),
		)
		step_2_pl = sp.make_pipeline(
			si.SimpleImputer(strategy="median"),
		)
		self.col_tf = sc.make_column_transformer(
			transformers=[
				("drop", drop_cols),
				(sd.PCA(n_components=1), temp_cols)
			],
			remainder="passthrough"
		)
		sc.ColumnTransformer().fit_transform

	def add_lags(self, df: pl.DataFrame):
		return (
			df.with_columns(
				pl.col("rainfall").shift(1).alias("rainfall_lag_1d"),
				pl.col("rainfall").shift(7).alias("rainfall_lag_1w"),
				pl.col("rainfall").shift(365).alias("rainfall_lag_1y"),
				pl.col("rainfall").shift(2 * 365).alias("rainfall_lag_2y"),
				pl.col("humidity").shift(1).alias("humidity_lag_1d"),
			)
		)

	def reduce_complexity(self):
		# combine
		pass

	def add_cyclicality(self, df: pl.DataFrame):
		pass

