import polars as pl
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import scipy.sparse as sp

side_col_expr: pl.Expr = (
	pl.when((pl.col("Deck") == "") | (pl.col("Number") == 0))
	.then(pl.lit("unknown"))
	.when(
		(pl.col("Deck") == "A") & (pl.col("Number") % 2 == 0) &
		(pl.col("Number").is_between(2, 40) | pl.col("Number").is_between(52, 98))
	).then(pl.lit("port"))
	.when(
		(pl.col("Deck") == "A") & (pl.col("Number") % 2 != 0) &
		(pl.col("Number").is_between(1, 37) | pl.col("Number").is_between(51, 97))
	).then(pl.lit("starboard"))
	.when(
		(pl.col("Deck") == "B") & (pl.col("Number") % 2 == 0) &
		(pl.col("Number").is_between(4, 42) | pl.col("Number").is_between(54, 92))
	).then(pl.lit("port"))
	.when(
		(pl.col("Deck") == "B") & (pl.col("Number") % 2 != 0) &
		(pl.col("Number").is_between(3, 39) | pl.col("Number").is_between(53, 91))
	).then(pl.lit("starboard"))
	.when(
		(pl.col("Deck") == "C") & (pl.col("Number") % 2 == 0) &
		(pl.col("Number").is_between(6, 44) | pl.col("Number").is_between(56, 96))
	).then(pl.lit("port"))
	.when(
		(pl.col("Deck") == "C") & (pl.col("Number") % 2 != 0) &
		(pl.col("Number").is_between(5, 41) | pl.col("Number").is_between(55, 95))
	).then(pl.lit("starboard"))
	.when(
		(pl.col("Deck") == "D") & (pl.col("Number") % 2 == 0) &
		(pl.col("Number").is_between(8, 46) | pl.col("Number").is_between(58, 98))
	).then(pl.lit("port"))
	.when(
		(pl.col("Deck") == "D") & (pl.col("Number") % 2 != 0) &
		(pl.col("Number").is_between(7, 43) | pl.col("Number").is_between(57, 97))
	).then(pl.lit("starboard"))
	.when(
		(pl.col("Deck") == "E") & (pl.col("Number") % 2 == 0) &
		(pl.col("Number").is_between(10, 48) | pl.col("Number").is_between(60, 100))
	).then(pl.lit("port"))
	.when(
		(pl.col("Deck") == "E") & (pl.col("Number") % 2 != 0) &
		(pl.col("Number").is_between(9, 45) | pl.col("Number").is_between(59, 99))
	).then(pl.lit("starboard"))
	.when(
		(pl.col("Deck") == "F") & (pl.col("Number") % 2 == 0) &
		(pl.col("Number").is_between(12, 50) | pl.col("Number").is_between(64, 104))
	).then(pl.lit("port"))
	.when(
		(pl.col("Deck") == "F") & (pl.col("Number") % 2 != 0) &
		(pl.col("Number").is_between(11, 47) | pl.col("Number").is_between(63, 103))
	).then(pl.lit("starboard"))
	.when(
		(pl.col("Deck") == "G") & (pl.col("Number") % 2 == 0) &
		(pl.col("Number").is_between(14, 52) | pl.col("Number").is_between(66, 106))
	).then(pl.lit("port"))
	.when(
		(pl.col("Deck") == "G") & (pl.col("Number") % 2 != 0) &
		(pl.col("Number").is_between(13, 49) | pl.col("Number").is_between(65, 105))
	).then(pl.lit("starboard"))
	.when(pl.col("Number").is_null())
	.then(pl.lit("unknown"))
	.otherwise(pl.lit("middle"))
	.cast(pl.Categorical).alias("Side"),
)

section_col_expr: pl.Expr = (
	pl.when((pl.col("Deck") == "") | (pl.col("Number") == 0))
	.then(pl.lit("unknown"))
	.when(
		((pl.col("Deck").is_in(["A", "B", "C", "D"])) & (pl.col("Number") <= 27)) |
		((pl.col("Deck").is_in(["E", "F", "G"])) & (pl.col("Number") <= 30))
	).then(pl.lit("front"))
	.when(
		((pl.col("Deck").is_in(["A", "B", "C", "D"])) & (pl.col("Number").is_between(28, 50))) |
		((pl.col("Deck").is_in(["E", "F", "G"])) & (pl.col("Number").is_between(31, 55)))
	).then(pl.lit("mid-front"))
	.when(
		((pl.col("Deck").is_in(["A", "B", "C", "D"])) & (pl.col("Number").is_between(51, 80))) |
		((pl.col("Deck").is_in(["E", "F", "G"])) & (pl.col("Number").is_between(56, 85)))
	).then(pl.lit("middle"))
	.when(
		((pl.col("Deck").is_in(["A", "B", "C", "D"])) & (pl.col("Number").is_between(81, 110))) |
		((pl.col("Deck").is_in(["E", "F", "G"])) & (pl.col("Number").is_between(86, 115)))
	).then(pl.lit("mid-back"))
	.otherwise(pl.lit("back"))
	.cast(pl.Categorical)
	.alias("Sections"),
)

inner_outer_col_expr: pl.Expr = (
	pl.when((pl.col("Deck") == "") | (pl.col("Number") == 0))
	.then(pl.lit("unknown"))
	.when(
		(pl.col("Deck") == "A") & (pl.col("Number").is_between(201, 220)) |
		(pl.col("Deck") == "B") & (pl.col("Number").is_between(221, 250)) |
		(pl.col("Deck") == "C") & (pl.col("Number").is_between(301, 340)) |
		(pl.col("Deck") == "D") & (pl.col("Number").is_between(401, 430)) |
		(pl.col("Deck") == "E") & (pl.col("Number").is_between(501, 540)) |
		(pl.col("Deck") == "F") & (pl.col("Number").is_between(601, 640)) |
		(pl.col("Deck") == "G") & (pl.col("Number").is_between(701, 750))
	).then(pl.lit("inner"))
	.when(
		(pl.col("Number").is_between(1, 45)) |
		(pl.col("Number").is_between(51, 100)) |
		(pl.col("Number").is_between(150, 200))
	).then(pl.lit("outer"))
	.otherwise(pl.lit("unknown"))
	.cast(pl.Categorical)
	.alias("CabinType")
)

class TitanicTransformer:
	def __init__(self):
		num_pipeline = Pipeline([
			("scaler", StandardScaler(with_mean=True))
		])

		cat_pipeline = Pipeline([
			("one_hot_encoder", OneHotEncoder(sparse_output=False)),
			("scaler", StandardScaler(with_mean=True))
		])

		num_cols = ["SibSp", "Parch"]
		cat_cols = ["Pclass", "Sex", "Embarked", "Deck", "Side", "Sections", "CabinType", "AgeCat"]

		self.col_transf = ColumnTransformer(
			[
				("num_pipeline", num_pipeline, num_cols),
				("cat_pipeline", cat_pipeline, cat_cols)
			],
			remainder="passthrough"
		)
		self.fitted = False
		self.added_y_col = False

	def transform(self, df: pl.DataFrame) -> (np.ndarray | sp.spmatrix):
		df = self.add_y_col_if_not_there(df)
		df = self.drop_cols(df)
		df = self.split_cabin_col_in_categories(df)
		df = self.age_col_to_categories(df)
		df = self.recast_cols(df)

		if self.fitted:
			data = self.col_transf.transform(df)
		else:
			self.fitted = True
			data = self.col_transf.fit_transform(df)

		if self.added_y_col:
			self.added_y_col = False
			return data[:, :-1]
		else:
			return data

	def add_y_col_if_not_there(self, df: pl.DataFrame):
		if not "Survived" in df.columns:
			self.added_y_col = True
			return df.with_columns(pl.Series(values=[None] * df.height).alias("Survived"))
		else:
			return df

	def drop_cols(self, df: pl.DataFrame) -> pl.DataFrame:
		return df.drop(
			["PassengerId", "Name", "Ticket", "Fare"],
			strict=False
		)

	def split_cabin_col_in_categories(self, df: pl.DataFrame) -> pl.DataFrame:
		return (
			df.with_columns(
				# Split the Cabin column and get the first element
				pl.col("Cabin").fill_null("").str.split(by=" ")
				.map_elements(lambda s: s[0] if not s.is_empty() else "", return_dtype=pl.String)
				.alias("FirstCabin")
			).with_columns(
				# Split into deck and number
				pl.col("FirstCabin").str.slice(0, 1).alias("Deck"),
				pl.col("FirstCabin").str.slice(1)
				.map_elements(lambda s: int(s) if s else 0, return_dtype=pl.Int16)
				.alias("Number")
			).drop(
				["Cabin", "FirstCabin"]
			).with_columns(side_col_expr)
			.with_columns(section_col_expr)
			.with_columns(inner_outer_col_expr)
			.with_columns(
				pl.col("Deck").map_elements(lambda s: "unknown" if not s else s, return_dtype=pl.String)
			)
			.drop(["Number"])
		)

	def age_col_to_categories(self, df: pl.DataFrame) -> pl.DataFrame:
		return (
			df.with_columns(
				pl.when(pl.col("Age").is_null() | pl.col("Age").is_nan())
				.then(pl.lit("unknown"))
				.when(pl.col("Age") <= 12)
				.then(pl.lit("child"))
				.when(pl.col("Age") <= 18)
				.then(pl.lit("teenager"))
				.when(pl.col("Age") <= 25)
				.then(pl.lit("young_adult"))
				.when(pl.col("Age") <= 50)
				.then(pl.lit("adult"))
				.when(pl.col("Age") <= 70)
				.then(pl.lit("old"))
				.otherwise(pl.lit("very_old"))
				.cast(pl.Categorical).alias("AgeCat")
			).drop(["Age"])
		)

	def recast_cols(self, df: pl.DataFrame) -> pl.DataFrame:
		return (
			df.with_columns(
				pl.col("Pclass").cast(pl.String).cast(pl.Categorical),
				pl.col("Sex").cast(pl.Categorical),
				pl.col("Embarked").cast(pl.Categorical),
				pl.col("Deck").cast(pl.Categorical),
			)
		)

