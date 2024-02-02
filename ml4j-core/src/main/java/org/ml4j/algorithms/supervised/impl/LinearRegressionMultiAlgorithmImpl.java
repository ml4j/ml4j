/*
 * Copyright 2014 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.algorithms.supervised.impl;

import org.ml4j.algorithms.impl.RegularizationContext;
import org.ml4j.algorithms.supervised.LinearRegressionMultiAlgorithm;
import org.ml4j.algorithms.supervised.LinearRegressionMultiHypothesisFunction;

import Jama.Matrix;

/**
* A linear regression multi-label algorithm using the normal equation, using regularization
* defined by the value of the regularizationLambda
*
* Warning - if specified regularisation lambda is zero, no regularisation
* with occur and there is a risk of matrix operations throwing a non-invertible
* exception.
*
* If regularisation is used, the necessary matrices are guaranteed to be invertible
*
* @author Michael Lavelle
*/
public class LinearRegressionMultiAlgorithmImpl implements LinearRegressionMultiAlgorithm<RegularizationContext> {


	private Matrix createFeatureMatrix(double[][] featureMatrixValues) {

		return new Matrix(featureMatrixValues);
	}
	

	private Matrix createLabelVectors(double[][] labelVectorValues) {
		/*
		 * Matrix labelVector = new
		 * Matrix(labelVectorValues.length,labelVectorValues.length); int i = 0;
		 * int r = 0; for (double[] values : labelVectorValues) { for (double
		 * label : values) { labelVector.set(i,r, label); i++; } r = r + 1; }
		 * return labelVector;
		 */
		Matrix labelVector = new Matrix(labelVectorValues);
		// return labelVector.transpose();
		return labelVector;
	}

	private Matrix createRegularisationMatrix(Matrix preRegularisationMatrix, double lambda) {
		Matrix regularisationMatrix = Matrix.identity(preRegularisationMatrix.getRowDimension(),
				preRegularisationMatrix.getColumnDimension());
		regularisationMatrix.set(0, 0, 0);
		regularisationMatrix.times(lambda);
		return regularisationMatrix;
	}

	@Override
	public LinearRegressionMultiHypothesisFunction getOptimalHypothesisFunction(double[][] x, double[][] y,RegularizationContext context) {

		double[][] thetas = new double[y[0].length][x[0].length];
		double regularisationLambda = context.getRegularizationLambda();
		Matrix featureMatrix = createFeatureMatrix(x);
		Matrix labelVector = createLabelVectors(y);

		Matrix result = featureMatrix.transpose().times(featureMatrix);
		if (regularisationLambda > 0) {
			// Regularise if lambda specified is greater than zero
			result = result.plus(createRegularisationMatrix(result, regularisationLambda));
		}
		result = result.inverse().times(featureMatrix.transpose());
		Matrix thetasMatrix = result.times(labelVector);

		for (int r = 0; r < y[0].length; r++) {
			for (int j = 0; j < thetas[r].length; j++) {
				thetas[r][j] = thetasMatrix.get(j, r);
			}
		}

		return new LinearRegressionMultiHypothesisFunction(thetas);
	}

	

}
