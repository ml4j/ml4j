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
import org.ml4j.algorithms.supervised.LinearRegressionAlgorithm;
import org.ml4j.algorithms.supervised.LinearRegressionHypothesisFunction;

import Jama.Matrix;
/**
* A linear regression single-numeric-label algorithm using the normal equation, using regularization
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
public class LinearRegressionAlgorithmImpl implements LinearRegressionAlgorithm<RegularizationContext> {

	
	private Matrix createFeatureMatrix(double[][] featureMatrixValues) {

		return new Matrix(featureMatrixValues);
	}
	
	

	private Matrix createLabelVector(double[] labelVectorValues) {
		Matrix labelVector = new Matrix(labelVectorValues.length, 1);
		int i = 0;
		for (double label : labelVectorValues) {
			labelVector.set(i, 0, label);
			i++;
		}
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
	public LinearRegressionHypothesisFunction getOptimalHypothesisFunction(double[][] x, double[] y,RegularizationContext regularizationContext) {

		double[] thetas = new double[x[0].length];
		double regularizationLambda = regularizationContext.getRegularizationLambda();
		Matrix featureMatrix = createFeatureMatrix(x);
		Matrix labelVector = createLabelVector(y);

		Matrix result = featureMatrix.transpose().times(featureMatrix);
		if (regularizationLambda > 0) {
			// Regularise if lambda specified is greater than zero
			result = result.plus(createRegularisationMatrix(result, regularizationLambda));
		}
		result = result.inverse().times(featureMatrix.transpose());
		Matrix thetasMatrix = result.times(labelVector);

		for (int j = 0; j < thetas.length; j++) {
			thetas[j] = thetasMatrix.get(j, 0);
		}
		return new LinearRegressionHypothesisFunction(thetas);
	}

	

}
