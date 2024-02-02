/*
 * Copyright 2020 the original author or authors.
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
package org.ml4j.nn.architectures.inception.inceptionv4.modules;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.architectures.inception.InceptionModuleDefinition;
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * @author Michael Lavelle
 */
public class InceptionADefinition implements InceptionModuleDefinition {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private InceptionV4WeightsLoader weightsLoader;
	private int inceptionAModuleIndex;
	private boolean withFreezeOut;

	public InceptionADefinition(InceptionV4WeightsLoader weightsLoader, int inceptionAModuleIndex) {
		this.inceptionAModuleIndex = inceptionAModuleIndex;
		this.weightsLoader = weightsLoader;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(35, 35, 384, false);
	}

	@Override
	public <T extends NeuralComponent<?>> InitialComponents3DGraphBuilder<T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start, NeuralComponentFactory<T> neuralComponentFactory) {
		
		int initialComponentIndex = inceptionAModuleIndex * 7 + 12;
		return start
				.withParallelPaths().withPath()
				.withConvolutionalAxons("conv2d_" + initialComponentIndex)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + initialComponentIndex + "_kernel0", 1, 1, 384, 96))
				.withFilterSize(1, 1).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons("batch_normalization_" + initialComponentIndex).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + initialComponentIndex + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + initialComponentIndex + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + initialComponentIndex + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction("relu_" + initialComponentIndex,
						ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath().withPath()
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 1))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 1) + "_kernel0", 1, 1, 384, 64))
				.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 1)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 1) + "_beta0", 64))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 1) + "_moving_mean0", 64))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 1) + "_moving_variance0", 64))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 1), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 2))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 2) + "_kernel0", 3, 3, 64, 96))
				.withFilterSize(3, 3).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 2)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 2) + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 2) + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 2) + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 2), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath().withPath()
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 3))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 3) + "_kernel0", 1, 1, 384, 64))
				.withFilterSize(1, 1).withFilterCount(64).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 3)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 3) + "_beta0", 64))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 3) + "_moving_mean0", 64))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 3) + "_moving_variance0", 64))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 64, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 3), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 4))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 4) + "_kernel0", 3, 3, 64, 96))
				.withFilterSize(3, 3).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 4)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 4) + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 4) + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 4) + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 4), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 5))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 5) + "_kernel0", 3, 3, 96, 96))
				.withFilterSize(3, 3).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 5)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 5) + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 5) + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 5) + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 5), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath().withPath()

				.withAveragePoolingAxons("average_pooling_1").withFilterSize(3, 3).withStride(1, 1).withSamePadding()
				.withConnectionToNeurons(new Neurons3D(35, 35, 384, false)).withConvolutionalAxons("conv2d_" + (initialComponentIndex + 6))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 6) + "_kernel0", 1, 1, 384, 96))
				.withFilterSize(1, 1).withFilterCount(96).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 6)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 6) + "_beta0", 96))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 6) + "_moving_mean0", 96))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 6) + "_moving_variance0", 96))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 96, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 6), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath()
				.endParallelPaths("inception_a_concat_" + inceptionAModuleIndex, PathCombinationStrategy.FILTER_CONCAT);
		
	}

	@Override
	public String getName() {
		return "inception_a_" + inceptionAModuleIndex;
	}

}
