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
 * 
 * @author Michael Lavelle
 * 
 *         A LinearRegressionMultiHypothesisFunction encapsulates a function h : double[] ->
 *         double[] so that predict(double[] numericFeatures) is a "good"
 *         predictor given an numeric features array for the corresponding values
 *         of a numeric label array.
 * 
 *         This LinearRegressionMultiHypothesisFunction uses a set of theta weights
 *         corresponding to the feature values to predict a value from a linear
 *         combination of the weighted values.
 */
public class LinearRegressionMultiHypothesisFunction implements HypothesisFunction<double[], double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private double[][] thetas;

	public LinearRegressionMultiHypothesisFunction(double[][] thetas) {
		this.thetas = thetas;
	}
	
	public double[][] getThetas()
	{
		return thetas;
	}

	@Override
	public double[] predict(double[] numericFeatures) {
		double[] ys = new double[thetas.length];
		for (int i = 0; i < ys.length; i++) {
			double y = 0d;
			for (int index = 0; index < numericFeatures.length; index++) {
				y = y + thetas[i][index] * numericFeatures[index];
			}
			ys[i] = y;
		}
		return ys;
	}

	
	public String toString() {
		String str = "";
		for (double[] thetas : this.thetas) {
			String s = "";
			for (int i = 0; i < thetas.length; i++) {
				s = s + "," + thetas[i];
			}
			str = str + s +  "\n";
		}
		return str;
	}

}
