package org.ml4j.nn.activationfunctions.base;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DifferentiableActivationFunctionTestBase extends TestBase {

	@Mock
	private NeuronsActivationContext context;

	@Mock
	private DifferentiableActivationFunctionActivation mockActivationFunctionActivation;

	@BeforeEach
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(context.getMatrixFactory()).thenReturn(matrixFactory);
	}

	@Test
	public void testConstruction() {

		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DifferentiableActivationFunction activationFunction = createDifferentiableActivationFunctionUnderTest(
				leftNeurons, rightNeurons);

		Assertions.assertNotNull(activationFunction);
		Assertions.assertNotNull(activationFunction.getActivationFunctionType());
		Assertions.assertNotNull(activationFunction.getActivationFunctionType());
	}

	@Test
	public void testActivation() {

		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DifferentiableActivationFunction activationFunction = createDifferentiableActivationFunctionUnderTest(
				leftNeurons, rightNeurons);

		Assertions.assertNotNull(activationFunction);

		NeuronsActivation inputActivation = createNeuronsActivation(100, 32);

		DifferentiableActivationFunctionActivation outputActivation = activationFunction.activate(inputActivation,
				context);

		Assertions.assertNotNull(outputActivation);

		Assertions.assertNotNull(outputActivation.getActivationFunction());

		Assertions.assertNotNull(outputActivation.getInput());

		Assertions.assertSame(inputActivation, outputActivation.getInput());

		Assertions.assertNotNull(outputActivation.getOutput());

		Assertions.assertEquals(32, outputActivation.getOutput().getExampleCount());
		Assertions.assertEquals(100, outputActivation.getOutput().getFeatureCount());

	}

	@Test
	public void testActivationGradient() {

		Neurons leftNeurons = new Neurons(100, false);
		Neurons rightNeurons = new Neurons(100, false);

		DifferentiableActivationFunction activationFunction = createDifferentiableActivationFunctionUnderTest(
				leftNeurons, rightNeurons);

		Assertions.assertNotNull(activationFunction);

		NeuronsActivation inputActivation = createNeuronsActivation(100, 32);
		NeuronsActivation outputActivation = createNeuronsActivation(100, 32);

		Mockito.when(mockActivationFunctionActivation.getInput()).thenReturn(inputActivation);
		Mockito.when(mockActivationFunctionActivation.getOutput()).thenReturn(outputActivation);

		NeuronsActivation gradientActivation = activationFunction.activationGradient(mockActivationFunctionActivation,
				context);

		Assertions.assertNotNull(gradientActivation);

		Assertions.assertEquals(32, gradientActivation.getExampleCount());
		Assertions.assertEquals(100, gradientActivation.getFeatureCount());

	}

	protected abstract DifferentiableActivationFunction createDifferentiableActivationFunctionUnderTest(
			Neurons leftNeurons, Neurons rightNeurons);

}
