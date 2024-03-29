/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.components.manytoone.base;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class ManyToOneDirectedComponentActivationTestBase extends TestBase {

	protected NeuronsActivation mockOutputActivation;

	@Mock
	protected AxonsContext mockAxonsContext;

	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;

	protected DirectedComponentGradient<NeuronsActivation> mockInboundGradient;

	@Mock
	protected ManyToOneDirectedComponent<?> mockManyToOneDirectedComponent;

	@Mock
	protected NeuronsActivation mockOutputGradientActivation;

	@BeforeEach
	public void setup() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);
		this.mockOutputActivation = createNeuronsActivation(110, 32);
		this.mockInboundGradient = MockTestData.mockComponentGradient(110, 32, this);
	}

	private ManyToOneDirectedComponentActivation createManyToOneDirectedComponentActivation(NeuronsActivation output,
			int inputCount) {
		return createManyToOneDirectedComponentActivationUnderTest(output, inputCount,
				PathCombinationStrategy.ADDITION);
	}

	protected abstract ManyToOneDirectedComponentActivation createManyToOneDirectedComponentActivationUnderTest(
			NeuronsActivation output, int inputCount, PathCombinationStrategy pathCombinationStrategy);

	@Test
	public void testConstruction() {
		ManyToOneDirectedComponentActivation manyToOneDirectedComponentActivation = createManyToOneDirectedComponentActivation(
				mockOutputActivation, 2);
		Assertions.assertNotNull(manyToOneDirectedComponentActivation);
	}

	@Test
	public void testGetOutput() {

		ManyToOneDirectedComponentActivation manyToOneDirectedComponentActivation = createManyToOneDirectedComponentActivation(
				mockOutputActivation, 2);
		Assertions.assertNotNull(manyToOneDirectedComponentActivation);
		Assertions.assertNotNull(manyToOneDirectedComponentActivation.getOutput());
		Assertions.assertSame(mockOutputActivation, manyToOneDirectedComponentActivation.getOutput());
	}

	@Test
	public void testBackPropagate() {

		ManyToOneDirectedComponentActivation manyToOneDirectedComponentActivation = createManyToOneDirectedComponentActivation(
				mockOutputActivation, 2);

		Assertions.assertNotNull(manyToOneDirectedComponentActivation);

		DirectedComponentGradient<List<NeuronsActivation>> backPropagatedGradient = manyToOneDirectedComponentActivation
				.backPropagate(mockInboundGradient);

		Assertions.assertNotNull(backPropagatedGradient);
		Assertions.assertNotNull(backPropagatedGradient.getOutput());
		Assertions.assertEquals(2, backPropagatedGradient.getOutput().size());
		Assertions.assertNotNull(backPropagatedGradient.getOutput().get(0));
		Assertions.assertNotNull(backPropagatedGradient.getOutput().get(1));

		Assertions.assertFalse(backPropagatedGradient.getOutput().get(0).getFeatureCount() == 0);
		Assertions.assertFalse(backPropagatedGradient.getOutput().get(0).getExampleCount() == 0);
		Assertions.assertFalse(backPropagatedGradient.getOutput().get(1).getFeatureCount() == 0);
		Assertions.assertFalse(backPropagatedGradient.getOutput().get(1).getExampleCount() == 0);

		Assertions.assertEquals(110, backPropagatedGradient.getOutput().get(0).getFeatureCount());
		Assertions.assertEquals(32, backPropagatedGradient.getOutput().get(0).getExampleCount());
		Assertions.assertEquals(110, backPropagatedGradient.getOutput().get(1).getFeatureCount());
		Assertions.assertEquals(32, backPropagatedGradient.getOutput().get(1).getExampleCount());

	}

}
