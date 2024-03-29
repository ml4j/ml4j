package org.ml4j.nn.activationfunctions;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.activationfunctions.base.DifferentiableActivationFunctionTestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

public class DefaultSoftmaxActivationFunctionImplTest extends DifferentiableActivationFunctionTestBase {

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
	protected DifferentiableActivationFunction createDifferentiableActivationFunctionUnderTest(Neurons leftNeurons,
			Neurons rightNeurons) {
		return new DefaultSoftmaxActivationFunctionImpl();
	}

	@Override
	@Test
	public void testActivationGradient() {

		Assertions.assertThrows(UnsupportedOperationException.class, () -> {


		super.testActivationGradient();

		});
	}

}
