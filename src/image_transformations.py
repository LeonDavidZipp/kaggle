import numpy as np
import skimage.transform as st
# import skimage.transform. as sf

class ImageTransformer:
	def __init__(self):
		pass

	@staticmethod
	def rotate(img: np.ndarray, deg: float) -> np.ndarray:
		return st.rotate(img, deg)

	@staticmethod
	def blur(img: np.ndarray) -> np.ndarray:
		return sf.gaussian(img, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)

	@staticmethod
	def resize_to(img: np.ndarray, row_target: int, col_target: int) -> np.ndarray:
		"""
		img: img in 2d format
		row_target:number of rows of new img
		col_target:number of cols of new img
		:returns: resized image (not as 1d array)
		"""
		return st.resize(img, (row_target, col_target))