import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

class MetricPlots():
	@staticmethod
	def gen_and_plot_conf_mx(y_true, y_pred):
		conf_mx = confusion_matrix(y_true, y_pred)
		plt.matshow(conf_mx, cmap=plt.cm.gray)
		plt.show()

	@staticmethod
	def gen_and_plot_error_conf_mx(y_true, y_pred):
		conf_mx = confusion_matrix(y_true, y_pred)
		row_sums = conf_mx.sum(axis=1, keepdims=True)
		norm_conf_mx = conf_mx / row_sums
		np.fill_diagonal(norm_conf_mx, 0)
		plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
		plt.show()

	@staticmethod
	def gen_and_plot_precision_and_recall(estimator: BaseEstimator, X, y):
		y_scores = cross_val_predict(estimator, X, y, cv=3, method="decision_function")
		precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
		plt.plot(thresholds, precisions[:-1], "b--", label="precision")
		plt.plot(thresholds, recalls[:-1], "r--", label="recall")
		plt.legend()
		plt.show()

	@staticmethod
	def gen_and_plot_roc_curve(y_true, y_scores):
		fpr, tpr, _ = roc_curve(y_true, y_scores)
		plt.plot(fpr, tpr, "b--")
		plt.plot([0, 1], [0, 1], "r--")
		plt.legend()
		plt.show()