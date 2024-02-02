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
public class InceptionCDefinition implements InceptionModuleDefinition {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private InceptionV4WeightsLoader weightsLoader;
	private int inceptionCModuleIndex;
	private boolean withFreezeOut;

	public InceptionCDefinition(InceptionV4WeightsLoader weightsLoader, int inceptionCModuleIndex) {
		this.inceptionCModuleIndex = inceptionCModuleIndex;
		this.weightsLoader = weightsLoader;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(8, 8, 1536, false);
	}

	@Override
	public <T extends NeuralComponent<?>> InitialComponents3DGraphBuilder<T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start, NeuralComponentFactory<T> neuralComponentFactory) {
		
		int initialComponentIndex = inceptionCModuleIndex * 10 + 120;
		return start
				.withParallelPaths()
				.withPath().withConvolutionalAxons("conv2d_" + initialComponentIndex)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + initialComponentIndex + "_kernel0", 1, 1, 1536, 256))
				.withFilterSize(1, 1).withFilterCount(256).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons("batch_normalization_" + initialComponentIndex).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + initialComponentIndex + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + initialComponentIndex + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + initialComponentIndex + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction("relu_" + initialComponentIndex, ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath().withPath()
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 1))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 1) + "_kernel0", 1, 1, 1536, 384))
				.withFilterSize(1, 1).withFilterCount(384).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 384, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 1)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 1) + "_beta0", 384))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 1) + "_moving_mean0", 384))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 1) + "_moving_variance0", 384))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 384, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 1), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).withParallelPaths()
				.withPath()
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 2))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 2) + "_kernel0", 3, 1, 384, 256))
				.withFilterSize(3, 1).withFilterCount(256).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 2)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 2) + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 2) + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 2) + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 2), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath().withPath()
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 3))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 3) + "_kernel0", 1, 3, 384, 256))
				.withFilterSize(1, 3).withFilterCount(256).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 3)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 3) + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 3) + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 3) + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 3), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath()
				.endParallelPaths("inception_c_concat_" + inceptionCModuleIndex + "_0", PathCombinationStrategy.FILTER_CONCAT).endPath().withPath()
				// 124
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 4))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 4) + "_kernel0", 1, 1, 1536, 384))
				.withFilterSize(1, 1).withFilterCount(384).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 384, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 4)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 4) + "_beta0", 384))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 4) + "_moving_mean0", 384))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 4) + "_moving_variance0", 384))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 384, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 4), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 5))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 5) + "_kernel0", 1, 3, 384, 448))
				.withFilterSize(1, 3).withFilterCount(448).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 448, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 5)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 5) + "_beta0", 448))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 5) + "_moving_mean0", 448))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 5) + "_moving_variance0", 448))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 448, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 5), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 6))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 6) + "_kernel0", 3, 1, 448, 512))
				.withFilterSize(3, 1).withFilterCount(512).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 512, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 6)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 6) + "_beta0", 512))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 6) + "_moving_mean0", 512))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 6) + "_moving_variance0", 512))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 512, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 6), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).withParallelPaths()
				.withPath()
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 7))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 7) + "_kernel0", 3, 1, 512, 256))
				.withFilterSize(3, 1).withFilterCount(256).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 7) ).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 7) + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 7) + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 7) + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 7) , ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath().withPath()
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 8))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 8) + "_kernel0", 1, 3, 512, 256))
				.withFilterSize(1, 3).withFilterCount(256).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 8)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 8) + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 8) + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 8) + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 8), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath()
				.endParallelPaths("inception_c_concat_" + inceptionCModuleIndex + "_1", PathCombinationStrategy.FILTER_CONCAT).endPath().withPath().withAveragePoolingAxons("average_pooling_4")
				.withFilterSize(3, 3).withStride(1, 1).withSamePadding()
				.withConnectionToNeurons(new Neurons3D(8, 8, 1536, false))
				.withConvolutionalAxons("conv2d_" + (initialComponentIndex + 9))
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights(
						"conv2d_" + (initialComponentIndex + 9) + "_kernel0", 1, 1, 1536, 256))
				.withFilterSize(1, 1).withFilterCount(256).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false)).withBatchNormAxons("batch_normalization_" + (initialComponentIndex + 9)).withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases(
						"batch_normalization_" + (initialComponentIndex + 9) + "_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerMean(
						"batch_normalization_" + (initialComponentIndex + 9) + "_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerVariance(
						"batch_normalization_" + (initialComponentIndex + 9) + "_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(8, 8, 256, false))
				.withActivationFunction("relu_" + (initialComponentIndex + 9), ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath()
				.endParallelPaths("inception_c_concat_" + inceptionCModuleIndex, PathCombinationStrategy.FILTER_CONCAT);
	}
	
	@Override
	public String getName() {
		return "inception_c_" + inceptionCModuleIndex;
	}

}
