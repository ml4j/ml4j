package org.ml4j.nn.axons.base;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;

public abstract class Axons3DTestBase<A extends Axons<?, ?, ?>> extends TestBase {

	private A axons;

	@Mock
	protected Neurons3D leftNeurons;

	@Mock
	protected Neurons3D rightNeurons;

	@Mock
	protected AxonsContext mockAxonsContext;

	protected NeuronsActivation mockLeftToRightInputActivation;

	public void setUp() {
		Axons3DConfig config = new Axons3DConfig(leftNeurons, rightNeurons);
		axons = createAxonsUnderTest(leftNeurons, rightNeurons, config);
	}

	protected abstract A createAxonsUnderTest(Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig config);

	protected abstract int getExpectedReformattedInputRows();

	protected abstract int getExpectedReformattedInputColumns();

	@Test
	public void testGetLeftNeurons() {
		Assertions.assertNotNull(axons.getLeftNeurons());
	}

	@Test
	public void testGetRightNeurons() {
		Assertions.assertNotNull(axons.getRightNeurons());
	}

	protected abstract boolean expectPostDropoutInputToBeSet();

	@Test
	public void testPushLeftToRight() {

		Mockito.when(mockAxonsContext.isTrainingContext()).thenReturn(true);
		
		Mockito.when(mockAxonsContext.getLeftHandInputDropoutKeepProbability()).thenReturn(1f);


		Mockito.when(mockAxonsContext.isTrainingContext()).thenReturn(true);

		AxonsActivation leftToRightActivation = axons.pushLeftToRight(mockLeftToRightInputActivation, null,
				mockAxonsContext);
		Assertions.assertNotNull(leftToRightActivation);
		if (expectPostDropoutInputToBeSet()) {
			NeuronsActivation postDropoutInput = leftToRightActivation.getPostDropoutInput().get();
			Assertions.assertNotNull(postDropoutInput);
			Assertions.assertEquals(getExpectedReformattedInputRows(), postDropoutInput.getFeatureCount());
			Assertions.assertEquals(getExpectedReformattedInputColumns(), postDropoutInput.getExampleCount());
			Assertions.assertEquals(mockLeftToRightInputActivation.getFeatureOrientation(),
					postDropoutInput.getFeatureOrientation());
		}

		NeuronsActivation postDropoutOutput = leftToRightActivation.getPostDropoutOutput();
		Assertions.assertNotNull(postDropoutOutput);

	}

}
