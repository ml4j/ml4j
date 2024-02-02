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

import org.ml4j.algorithms.impl.RegularizationContext;

/**
 * Learns a hypothesis function mapping numeric arrays to numeric label arrays using linear regression
 * 
 * @author Michael Lavelle
 *
 */
public interface LinearRegressionMultiAlgorithm<C extends RegularizationContext> {

	LinearRegressionMultiHypothesisFunction getOptimalHypothesisFunction(double[][] x, double[][] y,C context);

}
