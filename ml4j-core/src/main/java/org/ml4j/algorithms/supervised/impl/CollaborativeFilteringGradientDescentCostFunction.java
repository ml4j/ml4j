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

import org.ml4j.algorithms.impl.CostFunction;
import org.ml4j.algorithms.impl.GradientDescentCostFunction;
import org.ml4j.algorithms.supervised.CollaborativeFilteringHypothesisFunction;

import Jama.Matrix;

/**
 * Defines collaborative filtering cost function
 * 
 * @author Michael Lavelle
 */
public class CollaborativeFilteringGradientDescentCostFunction
		implements
		CostFunction<double[], double[], CollaborativeFilteringHypothesisFunction>,
		GradientDescentCostFunction<double[], double[], CollaborativeFilteringHypothesisFunction> {

	private double regularizationLambda;

	public CollaborativeFilteringGradientDescentCostFunction(int featureCount,
			double regularizationLambda) {
		this.regularizationLambda = regularizationLambda;
	}
	
	private Matrix createFeatureMatrix(double[][] featureMatrixValues) {

		return new Matrix(featureMatrixValues);
	}
	
	private double sum(Matrix matrix)
	{
		double sum = 0d;
		for (int i = 0; i < matrix.getRowDimension(); i++)
		{
			for (int j = 0; j < matrix.getColumnDimension(); j++)
			{
				sum = sum + matrix.get(i, j);
			}
				
		}
		return sum;
	}

	@Override
	public double getCost(CollaborativeFilteringHypothesisFunction h,
			double[][] features, double[][] labels) {
		int trainingExamples = features.length;
		double cost = 0d;
		
		Matrix R = createFeatureMatrix(features);
		Matrix X = createFeatureMatrix(h.getXs());
		Matrix THETA = createFeatureMatrix(h.getThetas());
		Matrix Y = createFeatureMatrix(labels);

		
		Matrix S = X.times(THETA.transpose()).minus(Y);

		Matrix result = S.arrayTimes(S).arrayTimes(R);
		cost = sum(result);
		

		cost = cost + regularizationLambda * (sum(THETA.arrayTimes(THETA)) + sum(X.arrayTimes(X)));
		cost = cost / 2;
		
		cost = cost / trainingExamples;

		return cost;
	}

	@Override
	public double[] getGradients(
			CollaborativeFilteringHypothesisFunction hypothesisFunction,
			double[][] featureMatrix, double[][] labelVector) {
		
		
		int thetaCount = hypothesisFunction.getThetas().length
				* hypothesisFunction.getThetas()[0].length;
		int xCount = hypothesisFunction.getXs().length
				* hypothesisFunction.getXs()[0].length;
		double[] gradients = new double[thetaCount + xCount];
		
		Matrix R = createFeatureMatrix(featureMatrix);
		Matrix X = createFeatureMatrix(hypothesisFunction.getXs());
		Matrix THETA = createFeatureMatrix(hypothesisFunction.getThetas());
		Matrix Y = createFeatureMatrix(labelVector);

		Matrix X_grad = X.times(THETA.transpose()).minus(Y).arrayTimes(R).times(THETA);
		Matrix THETA_grad = X.times(THETA.transpose()).minus(Y).arrayTimes(R).transpose().times(X);
		
		
		X_grad = X_grad.plus(X.times(regularizationLambda));
		
		THETA_grad = THETA_grad.plus(THETA.times(regularizationLambda));

		
		int ind = 0;

		for (int j = 0; j < THETA_grad.getRowDimension(); j++)
		{
			for (int f = 0; f < X.getColumnDimension(); f++)
			{
				gradients[ind++] = THETA_grad.get(j, f)/X.getRowDimension();
			}
		}
		
		
		for (int i = 0; i < X_grad.getRowDimension(); i++)
		{
			for (int f = 0; f < X.getColumnDimension(); f++)
			{
				gradients[ind++] = X_grad.get(i, f)/X.getRowDimension();
			}
		}


		return gradients;
	}

}
