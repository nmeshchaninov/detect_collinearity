"""
LinearRegression_R class
Replaces original fit function of parent class LinearRegression
to reproduce the comportment of lm function of R.
"""

import math

import numpy.linalg.linalg
import scipy
import statsmodels.api
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant


class LinearRegression_R(LinearRegression):

    def detect_collinear_columns(self, x):
        """
        Detects dependent features in matrix X
        Adds intercept vector at 0 position.

        Based on

        H. Engler "The Behavior of the QR-Factorization Algorithm
        with Column Pivoting" Department of Mathematics Georgetown University
        Washington, DC 20057, U.S.A. 1997:
        https://pages.stat.wisc.edu/~bwu62/771/engler1997.pdf

        LAPACK "QR Factorization with Column Pivoting":
        https://netlib.org/lapack/lug/node42.html

        C. Thiart "COLLINEARITY AND CONSEQUENCES FOR ESTIMATION:
        A STUDY AND SIMULATION", Department of Mathematical Statistics
        University of Cape Town, 1990

        https://open.uct.ac.za/bitstream/handle/11427/23615/thesis_sci_Thiart_1990.pdf

        Uses scipy QR decomposition function to retrieve pivot vector which
        gives an ordering of the columns by "most linearly independent".

        :param x: Matrix of features (variables)
        :return: array of zero-based indexes of dependent features (variables)
        """

        design_matrix = np.array(x)
        # add intercept column
        ones = np.ones((x.shape[0], 1))
        design_matrix = np.append(ones, design_matrix, axis=1)
        q, r, p = scipy.linalg.qr(design_matrix, pivoting=True, mode="economic")
        for i in range(r.shape[1]-1, 1, -1):
            if abs(r[i, i]) > 1e-7:
                x_rank = i+1
                break
        #x_rank = np.linalg.matrix_rank(design_matrix)
        vectors_num = design_matrix.shape[1]
        # check if matrix have a full rank
        if x_rank == vectors_num:
            return np.array([])


        # check if there are collinear vectors
        r2_cnt = vectors_num - x_rank
        r2 = p[-r2_cnt:]
        # recompute indexes to ignore intercept vector
        return r2 - 1

    def fit(self, x, y, sample_weight=None):
        """
        Overrides original fit method of LinearRegression class to
        avoid including collinear vector into the model

        :param x: Design matrix
        :param y: Vector of observed values
        :param sample_weight: Weights, passed directly to fit() function of superclass
        LinearRegression.
        :return: Nothing
        """
        r2 = self.detect_collinear_columns(x)
        r2_cnt = r2.shape[0]
        reduced_x = np.array(x)
        # zeroing dependent variables if any
        if r2_cnt > 0:
            reduced_x[:, r2] = 0

        super().fit(reduced_x, y, sample_weight)

        if r2_cnt > 0:
            for dep_col in r2:
                if len(self.coef_.shape) == 2:
                    self.coef_[:, dep_col] = None
                else:
                    self.coef_[dep_col] = None