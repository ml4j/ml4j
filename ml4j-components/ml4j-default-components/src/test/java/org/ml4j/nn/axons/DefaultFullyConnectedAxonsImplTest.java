package org.ml4j.nn.axons;

import java.util.Arrays;

import org.junit.jupiter.api.BeforeEach;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.base.AxonsTestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.mockito.Mockito;

public class DefaultFullyConnectedAxonsImplTest extends AxonsTestBase<FullyConnectedAxons> {

	@BeforeEach
	@Override
	public void setUp() {
		super.setUp();
	}

	@Override
	protected FullyConnectedAxons createAxonsUnderTest(Neurons leftNeurons, Neurons rightNeurons) {
		AxonWeights axonWeights = new FullyConnectedAxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(),
				rightNeurons.getNeuronCountExcludingBias(),
				new WeightsMatrixImpl(matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(),
						leftNeurons.getNeuronCountExcludingBias()),
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE),
						WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)),
				new BiasVectorImpl(matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(), 1), FeaturesVectorFormat.DEFAULT_BIAS_FORMAT), null);
		return new DefaultFullyConnectedAxonsImpl(new AxonsConfig<>(leftNeurons, rightNeurons), axonWeights);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return new NeuronsActivationImpl(new Neurons(featureCount, false),
				matrixFactory.createMatrix(featureCount, exampleCount),
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
	}

	@Override
	public void testPushLeftToRight() {
		Mockito.when(mockAxonsContext.getLeftHandInputDropoutKeepProbability()).thenReturn(0.5f);
		super.testPushLeftToRight();
	}

}
