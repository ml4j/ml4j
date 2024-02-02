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
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * @author Michael Lavelle
 */
public class ReductionADefinition implements Component3Dto3DGraphDefinition {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private InceptionV4WeightsLoader weightsLoader;
	private boolean withFreezeOut;

	public ReductionADefinition(InceptionV4WeightsLoader weightsLoader) {
		this.weightsLoader = weightsLoader;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(35, 35, 384, false);
	}

	@Override
	public Neurons3D getOutputNeurons() {
		return new Neurons3D(17, 17, 1024, false);
	}

	public <T extends NeuralComponent<?>> InitialComponents3DGraphBuilder<T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start, NeuralComponentFactory<T> neuralComponentFactory) {
		
		return start.withParallelPaths().withPath().withConvolutionalAxons("conv2d_40").withFilterSize(3, 3)
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_40_kernel0", 3, 3, 384, 384))
				.withStride(2, 2).withFilterCount(384).withValidPadding()
				.withAxonsContextConfigurer(
						context -> context.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false)).withBatchNormAxons("batch_normalization_40").withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_40_beta0", 384))
				.withMean(weightsLoader.getBatchNormLayerMean("batch_normalization_40_moving_mean0", 384))
				.withVariance(weightsLoader.getBatchNormLayerVariance("batch_normalization_40_moving_variance0", 384))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false))
				.withActivationFunction("relu_40", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath().withPath()
				.withConvolutionalAxons("conv2d_41")
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_41_kernel0", 1, 1, 384, 192))
				.withFilterSize(1, 1).withFilterCount(192).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 192, false)).withBatchNormAxons("batch_normalization_41").withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_41_beta0", 192))
				.withMean(weightsLoader.getBatchNormLayerMean("batch_normalization_41_moving_mean0", 192))
				.withVariance(weightsLoader.getBatchNormLayerVariance("batch_normalization_41_moving_variance0", 192))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 192, false))
				.withActivationFunction("relu_41", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.withConvolutionalAxons("conv2d_42")
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_42_kernel0", 3, 3, 192, 224))
				.withFilterSize(3, 3).withFilterCount(224).withSamePadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 224, false)).withBatchNormAxons("batch_normalization_42").withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_42_beta0", 224))
				.withMean(weightsLoader.getBatchNormLayerMean("batch_normalization_42_moving_mean0", 224))
				.withVariance(weightsLoader.getBatchNormLayerVariance("batch_normalization_42_moving_variance0", 224))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(35, 35, 224, false))
				.withActivationFunction("relu_42", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties())
				.withConvolutionalAxons("conv2d_43")
				.withConnectionWeights(weightsLoader.getConvolutionalLayerWeights("conv2d_43_kernel0", 3, 3, 224, 256))
				.withStride(2, 2).withFilterSize(3, 3).withFilterCount(256).withValidPadding()
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false)).withBatchNormAxons("batch_normalization_43").withBiasUnit()
				.withBeta(weightsLoader.getBatchNormLayerBiases("batch_normalization_43_beta0", 256))
				.withMean(weightsLoader.getBatchNormLayerMean("batch_normalization_43_moving_mean0", 256))
				.withVariance(weightsLoader.getBatchNormLayerVariance("batch_normalization_43_moving_variance0", 256))
				.withAxonsContextConfigurer(
						c -> c.withFreezeOut(withFreezeOut))
				.withConnectionToNeurons(new Neurons3D(17, 17, 256, false))
				.withActivationFunction("relu_43", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), new ActivationFunctionProperties()).endPath().withPath()
				.withMaxPoolingAxons("max_pooling_3").withFilterSize(3, 3).withStride(2, 2).withValidPadding()
				.withConnectionToNeurons(new Neurons3D(17, 17, 384, false)).endPath()
				.endParallelPaths("reduction_a_concat_0", PathCombinationStrategy.FILTER_CONCAT);
	}

	@Override
	public String getName() {
		return "reduction_a";
	}
}
