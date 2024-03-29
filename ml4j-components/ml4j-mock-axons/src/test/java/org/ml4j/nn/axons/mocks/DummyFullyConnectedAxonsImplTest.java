package org.ml4j.nn.axons.mocks;

import org.junit.jupiter.api.BeforeEach;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.base.AxonsTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyFullyConnectedAxonsImplTest extends AxonsTestBase<FullyConnectedAxons> {

	private MatrixFactory matrixFactory;

	@BeforeEach
	@Override
	public void setUp() {
		matrixFactory = new JBlasRowMajorMatrixFactory();
		super.setUp();
	}

	@Override
	protected FullyConnectedAxons createAxonsUnderTest(Neurons leftNeurons, Neurons rightNeurons) {
		return new DummyFullyConnectedAxonsImpl(matrixFactory, leftNeurons, rightNeurons);
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount);
	}

}
