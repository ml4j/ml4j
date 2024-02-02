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
package org.ml4j.nn.components.manytomany.base;

import java.util.Arrays;
import java.util.List;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatchActivation;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DefaultDirectedComponentChainBatchActivationTestBase extends TestBase {

	@Mock
	protected AxonsContext mockAxonsContext;

	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;

	@Mock
	protected DefaultDirectedComponentChainActivation chainActivation1;

	@Mock
	protected DefaultDirectedComponentChainActivation chainActivation2;

	@Mock
	protected ManyToOneDirectedComponent<?> mockManyToOneDirectedComponent;

	@BeforeEach
	public void setup() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);
	}

	private DefaultDirectedComponentChainBatchActivation createDefaultDirectedComponentChainBatchActivation(
			List<DefaultDirectedComponentChainActivation> activations) {
		return createDefaultDirectedComponentChainBatchActivationUnderTest(activations);
	}

	protected abstract DefaultDirectedComponentChainBatchActivation createDefaultDirectedComponentChainBatchActivationUnderTest(
			List<DefaultDirectedComponentChainActivation> activations);

	@Test
	public void testConstruction() {
		DefaultDirectedComponentChainBatchActivation chainBatchActivation = createDefaultDirectedComponentChainBatchActivation(
				Arrays.asList(chainActivation1, chainActivation2));
		Assertions.assertNotNull(chainBatchActivation);
	}

	@Test
	public void testGetOutput() {

		NeuronsActivation mockOutputActivation1 = MockTestData.mockNeuronsActivation(110, 32);

		NeuronsActivation mockOutputActivation2 = MockTestData.mockNeuronsActivation(110, 32);

		Mockito.when(chainActivation1.getOutput()).thenReturn(mockOutputActivation1);
		Mockito.when(chainActivation2.getOutput()).thenReturn(mockOutputActivation2);

		DefaultDirectedComponentChainBatchActivation chainBatchActivation = createDefaultDirectedComponentChainBatchActivation(
				Arrays.asList(chainActivation1, chainActivation2));
		Assertions.assertNotNull(chainBatchActivation);
		Assertions.assertNotNull(chainBatchActivation.getOutput());
		Assertions.assertEquals(2, chainBatchActivation.getOutput().size());
		Assertions.assertNotNull(chainBatchActivation.getOutput().get(0));
		Assertions.assertNotNull(chainBatchActivation.getOutput().get(1));
		Assertions.assertSame(mockOutputActivation1, chainBatchActivation.getOutput().get(0));
		Assertions.assertSame(mockOutputActivation2, chainBatchActivation.getOutput().get(1));

	}

	@Test
	public void testBackPropagate() {

		NeuronsActivation mockOutputActivation1 = MockTestData.mockNeuronsActivation(110, 32);

		NeuronsActivation mockOutputActivation2 = MockTestData.mockNeuronsActivation(110, 32);

		Mockito.when(chainActivation1.getOutput()).thenReturn(mockOutputActivation1);
		Mockito.when(chainActivation2.getOutput()).thenReturn(mockOutputActivation2);

		DirectedComponentGradient<List<NeuronsActivation>> mockInboundGradient = MockTestData
				.mockBatchComponentGradient(110, 32, 2);

		DirectedComponentGradient<NeuronsActivation> outputGradient1 = MockTestData.mockComponentGradient(110, 32,
				this);
		DirectedComponentGradient<NeuronsActivation> outputGradient2 = MockTestData.mockComponentGradient(110, 32,
				this);

		Mockito.when(chainActivation1.backPropagate(Mockito.any())).thenReturn(outputGradient1);
		Mockito.when(chainActivation2.backPropagate(Mockito.any())).thenReturn(outputGradient2);
		// TODO verify

		DefaultDirectedComponentChainBatchActivation chainBatchActivation = createDefaultDirectedComponentChainBatchActivation(
				Arrays.asList(chainActivation1, chainActivation2));
		Assertions.assertNotNull(chainBatchActivation);

		DirectedComponentGradient<List<NeuronsActivation>> backPropagatedGradient = chainBatchActivation
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

		Assertions.assertEquals(mockInboundGradient.getOutput().get(0).getFeatureCount(),
				backPropagatedGradient.getOutput().get(0).getFeatureCount());
		Assertions.assertEquals(mockInboundGradient.getOutput().get(0).getExampleCount(),
				backPropagatedGradient.getOutput().get(0).getExampleCount());
		Assertions.assertEquals(mockInboundGradient.getOutput().get(1).getFeatureCount(),
				backPropagatedGradient.getOutput().get(1).getFeatureCount());
		Assertions.assertEquals(mockInboundGradient.getOutput().get(1).getExampleCount(),
				backPropagatedGradient.getOutput().get(1).getExampleCount());

	}

}
