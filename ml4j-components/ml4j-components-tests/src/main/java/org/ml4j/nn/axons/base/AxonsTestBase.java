package org.ml4j.nn.axons.base;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class AxonsTestBase<A extends Axons<?, ?, ?>> extends TestBase {

	private A axons;

	@Mock
	protected Neurons leftNeurons;

	@Mock
	protected Neurons rightNeurons;

	@Mock
	protected AxonsContext mockAxonsContext;

	protected MatrixFactory matrixFactory;

	@BeforeEach
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		this.matrixFactory = createMatrixFactory();
		Mockito.when(leftNeurons.getNeuronCountExcludingBias()).thenReturn(100);
		Mockito.when(rightNeurons.getNeuronCountExcludingBias()).thenReturn(110);
		Mockito.when(mockAxonsContext.getMatrixFactory()).thenReturn(matrixFactory);
		Mockito.when(mockAxonsContext.getLeftHandInputDropoutKeepProbability()).thenReturn(1f);

		axons = createAxonsUnderTest(leftNeurons, rightNeurons);
	}

	protected abstract A createAxonsUnderTest(Neurons leftNeurons, Neurons rightNeurons);

	@Test
	public void testGetLeftNeurons() {
		Neurons leftNeurons = axons.getLeftNeurons();
		Assertions.assertNotNull(leftNeurons);
	}

	@Test
	public void testGetRightNeurons() {
		Neurons leftNeurons = axons.getRightNeurons();
		Assertions.assertNotNull(leftNeurons);
	}

	@Test
	public void testPushLeftToRight() {

		NeuronsActivation mockLeftToRightInputActivation = createNeuronsActivation(100, 32);

		Mockito.when(mockAxonsContext.isTrainingContext()).thenReturn(true);

		AxonsActivation leftToRightActivation = axons.pushLeftToRight(mockLeftToRightInputActivation, null,
				mockAxonsContext);
		Assertions.assertNotNull(leftToRightActivation);
		NeuronsActivation postDropoutInput = leftToRightActivation.getPostDropoutInput().get();
		Assertions.assertNotNull(postDropoutInput);
		Assertions.assertEquals(100, postDropoutInput.getFeatureCount());
		Assertions.assertEquals(32, postDropoutInput.getExampleCount());
		Assertions.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET,
				postDropoutInput.getFeatureOrientation());

		NeuronsActivation postDropoutOutput = leftToRightActivation.getPostDropoutOutput();
		Assertions.assertNotNull(postDropoutOutput);

	}

}
