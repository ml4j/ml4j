/*
 * Copyright 2019 the original author or authors.
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
package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Encapsulates reformatting logic for a Convolutional component.
 * 
 * @author Michael Lavelle
 */
public interface ConvolutionalFormatter {

	Matrix reformatLeftToRightInput(MatrixFactory matrixFactory, NeuronsActivation activations);

	Matrix reformatRightToLeftOutput(MatrixFactory matrixFactory, Matrix initialOutputMatrix);

	Matrix reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation activations1);

	Matrix reformatLeftToRightOutput(MatrixFactory matrixFactory, Matrix origOutput);

	Matrix getIndexes();

}
