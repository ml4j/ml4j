package org.ml4j.nn.activationfunctions.base;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DifferentiableActivationFunctionActivationTestBase extends TestBase {

	@Mock
	private NeuronsActivationContext context;

	@Mock
	private DifferentiableActivationFunction mockActivationFunction;

	@BeforeEach
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(context.getMatrixFactory()).thenReturn(matrixFactory);
	}

	@Test
	public void testConstruction() {

		NeuronsActivation input = createNeuronsActivation(100, 32);
		NeuronsActivation output = createNeuronsActivation(100, 32);

		DifferentiableActivationFunctionActivation activationFunctionActivation = createDifferentiableActivationFunctionActivationUnderTest(
				mockActivationFunction, input, output);

		Assertions.assertNotNull(activationFunctionActivation);
		Assertions.assertNotNull(activationFunctionActivation.getActivationFunction());
		Assertions.assertSame(mockActivationFunction, activationFunctionActivation.getActivationFunction());
		Assertions.assertSame(input, activationFunctionActivation.getInput());
		Assertions.assertSame(output, activationFunctionActivation.getOutput());

	}

	protected abstract DifferentiableActivationFunctionActivation createDifferentiableActivationFunctionActivationUnderTest(
			DifferentiableActivationFunction activationFunction, NeuronsActivation input, NeuronsActivation output);

}
