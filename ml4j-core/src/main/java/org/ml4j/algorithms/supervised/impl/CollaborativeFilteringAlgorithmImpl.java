package org.ml4j.algorithms.supervised.impl;

import org.ml4j.algorithms.impl.GradientDescentAlgorithmImpl;
import org.ml4j.algorithms.impl.GradientDescentAlgorithmTrainingContext;
import org.ml4j.algorithms.supervised.CollaborativeFilteringAlgorithm;
import org.ml4j.algorithms.supervised.CollaborativeFilteringHypothesisFunction;


public class CollaborativeFilteringAlgorithmImpl extends
		GradientDescentAlgorithmImpl<double[], double[]>
		implements
		CollaborativeFilteringAlgorithm<GradientDescentAlgorithmTrainingContext> {

	private int featureCount;

	public CollaborativeFilteringAlgorithmImpl(int featureCount) {
		this.featureCount = featureCount;
	}
	
	
	private double[][] getRatingFlags(Double[][] ratings) 
	{
		
		double[][] ratingFlags = new double[ratings.length][ratings[0].length];
		for (int i = 0; i < ratings.length; i++)
		{
			for (int j = 0; j < ratings[0].length; j++)
			{
				ratingFlags[i][j] = ratings[i][j] != null ? 1 : 0;
			}
		}
		return ratingFlags;
	}
	
	private double[][] getRatingMatrix(Double[][] ratings)
	{
		
		double[][] ratingMatrix = new double[ratings.length][ratings[0].length];
		for (int i = 0; i < ratings.length; i++)
		{
			for (int j = 0; j < ratings[0].length; j++)
			{
				ratingMatrix[i][j] = ratings[i][j] != null ? ratings[i][j].doubleValue() : 0;
			}
		}
		return ratingMatrix;
	}

	@Override
	protected double[] getGradients(double[][] featureMatrix,
			double[][] labelVector, double[] thetas,
			GradientDescentAlgorithmTrainingContext trainingContext) {
		int userCount = featureMatrix[0].length;
		int itemCount = featureMatrix.length;
		return new CollaborativeFilteringGradientDescentCostFunction(2,
				trainingContext.getRegularizationLambda()).getGradients(
				createHypothesisFunction(thetas, userCount, itemCount),
				featureMatrix, labelVector);
	}

	public CollaborativeFilteringHypothesisFunction getOptimalHypothesisFunction(Double[][] y,
			GradientDescentAlgorithmTrainingContext context) {
		int userCount = y[0].length;
		int itemCount = y.length;
		return createHypothesisFunction(getOptimalThetas(getRatingFlags(y), getRatingMatrix(y), context),
				userCount, itemCount);
	}

	protected CollaborativeFilteringHypothesisFunction createHypothesisFunction(
			double[] gradientDescentThetas, int userCount, int itemCount) {

		double[][] thetas = new double[userCount][featureCount];
		double[][] xs = new double[itemCount][featureCount];
		int ind = 0;
		for (int u = 0; u < userCount; u++) {
			for (int f = 0; f < featureCount; f++) {
				thetas[u][f] = gradientDescentThetas[ind++];
			}
		}

		for (int i = 0; i < itemCount; i++) {
			for (int f = 0; f < featureCount; f++) {
				xs[i][f] = gradientDescentThetas[ind++];
			}
		}
		return new CollaborativeFilteringHypothesisFunction(thetas, xs);
	}

	@Override
	protected double[] getInitialThetas(double[][] x, double[][] y) {

		int userCount = x[0].length;
		int itemCount = x.length;

		double[] initialThetas = new double[userCount * featureCount
				+ itemCount * featureCount];
		for (int i = 0; i < initialThetas.length; i++) {
			initialThetas[i] = Math.random() * 0.01d;
		}
		return initialThetas;
	}

	@Override
	protected double getCost(double[] thetas, double[][] x, double[][] y,
			GradientDescentAlgorithmTrainingContext trainingContext) {

		int userCount = x[0].length;
		int itemCount = x.length;
		return new CollaborativeFilteringGradientDescentCostFunction(2,
				trainingContext.getRegularizationLambda()).getCost(
				createHypothesisFunction(thetas, userCount, itemCount), x, y);
	}

}
