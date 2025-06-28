import polars as pl
import numpy as np
import sklearn.base as sb
import sklearn.decomposition as sd
import sklearn.pipeline as sp
import sklearn.model_selection as sm
import sklearn.preprocessing as spp
import sklearn.compose as sc
import sklearn.impute as si
import statsmodels as sm
import feature_engine.creation as fc


class RainfallTransformer(sb.BaseEstimator):
	def __init__(self):
		drop_cols = ["dewpoint"]
		temp_cols = ["temparature", "mintemp", "maxtemp"]
		self.impute_scale_pl = sp.make_pipeline(
			si.SimpleImputer(strategy="median"),
			spp.MinMaxScaler(),
		)
		self.col_tf: sc.ColumnTransformer = sc.make_column_transformer(
			("drop", drop_cols),
			(sd.PCA(n_components=1), temp_cols),
			remainder="passthrough"
		)
		self.cyc_tf: fc.CyclicalFeatures = fc.CyclicalFeatures(
			variables=["day"],
			drop_original=True
		)

		self.col_tf_fitted = False
		self.cyc_tf_fitted = False
		self.added_y_col = False

	def cyclical_transform_cols(self, df: pl.DataFrame):
		if self.col_tf_fitted:
			data = self.cyc_tf.transform(df.to_pandas())
		else:
			self.cyc_tf_fitted = True
			data = self.cyc_tf.fit_transform(df.to_pandas())

		df = pl.DataFrame(data)
		return (
			df.select(
				pl.exclude("rainfall"),
				pl.col("rainfall")
			)
		)

	def transform(self, df: pl.DataFrame):
		if not "rainfall" in df.columns:
			self.added_y_col = True
			df = df.with_columns(pl.Series(values=[None] * df.height).alias("rainfall"))

		df = self.cyclical_transform_cols(df)
		print(df.head())

		if self.col_tf_fitted:
			data = self.col_tf.transform(df)
			data_tf = self.impute_scale_pl.transform(data[:, :-1])
			last_col = data[:, -1].reshape(-1, 1)
			result = np.hstack([data_tf, last_col])
		else:
			self.col_tf_fitted = True
			data = self.col_tf.fit_transform(df)
			data_tf = self.impute_scale_pl.fit_transform(data[:, :-1])
			last_col = data[:, -1].reshape(-1, 1)
			result = np.hstack([data_tf, last_col])

		if self.added_y_col:
			self.added_y_col = False
			return result[:, :-1]
		else:
			return result
