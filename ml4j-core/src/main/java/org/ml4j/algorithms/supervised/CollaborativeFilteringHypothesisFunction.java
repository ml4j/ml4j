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
package org.ml4j.algorithms.supervised;

import org.ml4j.algorithms.HypothesisFunction;

/**
 * Predicts all users' ratings for a given item, given (learned) item features.
 * The learned item features are contained within the rows of the xs matrix, and
 * the learned user preferences are contained in the rows of the thetas matrix
 * 
 * 
 * @author Michael Lavelle
 * 
 *
 */
public class CollaborativeFilteringHypothesisFunction implements
		HypothesisFunction<double[], double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private double[][] thetas;
	private double[][] xs;

	public double[][] getThetas() {
		return thetas;
	}

	public double[][] getXs() {
		return xs;
	}

	public CollaborativeFilteringHypothesisFunction(double[][] thetas,
			double[][] xs) {
		this.thetas = thetas;
		this.xs = xs;
	}

	@Override
	public double[] predict(double[] numericFeatures) {
		double[] y = new double[thetas.length];
		for (int k = 0; k < y.length; k++) {
			y[k] = 0d;
		}
		for (int u = 0; u < thetas.length; u++) {
			for (int index = 0; index < numericFeatures.length; index++) {
				y[u] = y[u] + thetas[u][index] * numericFeatures[index];
			}
		}
		return y;
	}

	public String toString() {

		String s = "";
		for (int i = 0; i < thetas.length; i++) {
			s = s + "," + thetas[i];
		}
		for (int i = 0; i < xs.length; i++) {
			s = s + "," + xs[i];
		}
		return s;
	}

}
