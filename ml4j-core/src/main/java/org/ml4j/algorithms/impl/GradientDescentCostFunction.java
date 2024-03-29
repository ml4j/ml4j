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
package org.ml4j.algorithms.impl;

import org.ml4j.algorithms.HypothesisFunction;


/**
 * 
 * @param <X> the type of an input data element which contains the set of independent attributes
 * @param <Y> the type of the dependent output or target data attribute, which the associated hypothesis function predicts, given an element of X 
 * @param <H> the type of the hypothesis function used to predict Y from X 

 * @author Michael Lavelle
 * 
 * A GradientFunction<X,Y,H> calculates the gradients associated with the cost function of training prediction errors using a given HypothesisFunction mapping elements of 
 * type X to type Y.   Each gradient is the partial derivative with respect to the relevant parameter theta.
 */
public interface GradientDescentCostFunction<X,Y,H extends HypothesisFunction<X,Y>> extends CostFunction<X,Y,H> {

	public double[] getGradients(H h,X[] x,Y[] y);
}