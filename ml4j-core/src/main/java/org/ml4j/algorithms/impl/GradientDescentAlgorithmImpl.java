package org.ml4j.algorithms.impl;


public abstract class GradientDescentAlgorithmImpl<X,Y>  {
	
	protected abstract double[] getGradients(X[] featureMatrix, Y[] labelVector,
			double[] thetas,GradientDescentAlgorithmTrainingContext trainingContext);
	
	protected double[] performThetasUpdateIteration(
			X[] featureMatrix, Y[] labelVector,
			double[] thetas,
			GradientDescentAlgorithmTrainingContext trainingContext) {

		double[] oldThetas = thetas;
		double[] newThetas = new double[oldThetas.length];		
		double[] gradients = getGradients(featureMatrix,labelVector,thetas,trainingContext);		
		for (int i = 0; i < gradients.length; i++)
		{
			newThetas[i] = oldThetas[i] - trainingContext.getLearningRateAlpha() * gradients[i];
		}
		return newThetas;
	}

	public double[] getOptimalThetas(
			X[] x, Y[] y,GradientDescentAlgorithmTrainingContext trainingContext) {
		double[] thetas = getInitialThetas(x,y);
			
		boolean snapshotTakenOfLastIteration = false;
		
		if (trainingContext.getLearningRateAlpha() == null)
		{
			throw new RuntimeException("No learning rate alpha specified on training context");
		}
		
		while (trainingContext.isTrainingRunning()
				&& !trainingContext.isTrainingSuccessful()) {
			snapshotTakenOfLastIteration = takeSnapshotOfCostFunctionValueIfApplicable(
					x, y, trainingContext,
					thetas);

			trainingContext.incrementIterationNumber();

			thetas = performThetasUpdateIteration(
					x, y, thetas,
					trainingContext);

		}

		if (!snapshotTakenOfLastIteration) {
			takeSnapshotOfCostFunctionValueIfApplicable(x,
					y, trainingContext, thetas);
		}

		if (trainingContext.isTrainingSuccessful()) {
			GradientDescentAlgorithmTrainingContext resetTrainingContext = new GradientDescentAlgorithmTrainingContext(trainingContext.getMaxIterations());
			resetTrainingContext.setLearningRateAlpha(trainingContext.getLearningRateAlpha());
			resetTrainingContext.setConvergenceCriteria(trainingContext.getConvergenceCriteria());
			resetTrainingContext.setRegularizationLambda(trainingContext.getRegularizationLambda());
			resetTrainingContext.setCostFunctionSnapshotIntervalInIterations(trainingContext.getCostFunctionSnapshotIntervalInIterations());
			trainingContext = resetTrainingContext;
			
			
			return thetas;
		} else {
			if (trainingContext.getConvergenceCriteria() != null) {
				throw new RuntimeException(
						"Training has stopped running but has not satified convergence criteria");
			} else {
				throw new RuntimeException(
						"Training has stopped running but cannot be deemed to have converged as no convergence criteria have been specified on the training context");
			}
		}

	}


	protected abstract double[] getInitialThetas(X[] x,
			Y[] y);


	protected abstract double getCost(double[] thetas,X[] x,
			Y[] y,GradientDescentAlgorithmTrainingContext trainingContext);
	
	private boolean takeSnapshotOfCostFunctionValueIfApplicable(X[] x,
			Y[] y,
			GradientDescentAlgorithmTrainingContext trainingContext,
			double[] thetas) {
		if (trainingContext.getCostFunctionSnapshotIntervalInIterations() != null && trainingContext.getCurrentIteration()
				% trainingContext.getCostFunctionSnapshotIntervalInIterations() == 0) {
			
			double cost = getCost(thetas,x,y,trainingContext);
			trainingContext.addCostFunctionSnapshotValue(cost);
			return true;
		}
		return false;		
	}



}
