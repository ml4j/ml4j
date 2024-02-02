package org.ml4j.algoritms.unsupervised;

import java.util.Map;

public interface ClusteringAlgorithm {

	double[][] getCentroids(double[][] data, int numberOfCentroids);

	Map<double[], double[][]> getClusters(double[][] trainingSet,
			double[][] centroids);

}
