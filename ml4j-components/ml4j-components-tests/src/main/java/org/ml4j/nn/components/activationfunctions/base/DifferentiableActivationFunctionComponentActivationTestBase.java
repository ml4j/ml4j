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
package org.ml4j.nn.components.activationfunctions.base;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public abstract class DifferentiableActivationFunctionComponentActivationTestBase<L extends DifferentiableActivationFunctionComponent>
		extends TestBase {

	@Mock
	protected AxonsContext mockAxonsContext;

	protected NeuronsActivation mockInputActivation;

	protected NeuronsActivation mockOutputActivation;

	protected DirectedComponentGradient<NeuronsActivation> mockInboundGradient;

	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;

	protected L activationFunctionComponent;

	// TODO THUR
	// @Mock
	// protected DifferentiableActivationFunction mockActivationFunction;

	@Mock
	protected ManyToOneDirectedComponent<?> mockManyToOneDirectedComponent;

	@BeforeEach
	public void setup() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(createMatrixFactory());
		// Mockito.when(mockActivationFunctionComponent.getActivationFunction()).thenReturn(mockActivationFunction);
		this.mockInputActivation = createNeuronsActivation(110, 32);
		this.mockOutputActivation = createNeuronsActivation(110, 32);
		this.activationFunctionComponent = createMockActivationFunctionComponent();
		this.mockInboundGradient = MockTestData.mockComponentGradient(110, 32, this);
	}

	private DifferentiableActivationFunctionComponentActivation createDifferentiableActivationFunctionComponentActivation(
			L activationFunction, NeuronsActivation input, NeuronsActivation output) {
		return createDifferentiableActivationFunctionComponentActivationUnderTest(activationFunction, input, output);
	}

	protected abstract L createMockActivationFunctionComponent();

	protected abstract DifferentiableActivationFunctionComponentActivation createDifferentiableActivationFunctionComponentActivationUnderTest(
			L activationFunction, NeuronsActivation input, NeuronsActivation output);

	@Test
	public void testConstruction() {
		DifferentiableActivationFunctionComponentActivation activation = createDifferentiableActivationFunctionComponentActivation(
				activationFunctionComponent, mockInputActivation, mockOutputActivation);
		Assertions.assertNotNull(activation);
	}

	@Test
	public void testGetOutput() {
		DifferentiableActivationFunctionComponentActivation activation = createDifferentiableActivationFunctionComponentActivation(
				activationFunctionComponent, mockInputActivation, mockOutputActivation);
		Assertions.assertNotNull(activation);
		Assertions.assertNotNull(activation.getOutput());
		Assertions.assertSame(mockOutputActivation, activation.getOutput());
	}

	@Test
	public void testBackPropagate() {

		DifferentiableActivationFunctionComponentActivation activation = createDifferentiableActivationFunctionComponentActivation(
				activationFunctionComponent, mockInputActivation, mockOutputActivation);
		Assertions.assertNotNull(activation);

		DirectedComponentGradient<NeuronsActivation> backPropagatedGradient = activation
				.backPropagate(mockInboundGradient);

		Assertions.assertNotNull(backPropagatedGradient);
		Assertions.assertNotNull(backPropagatedGradient.getOutput());
		Assertions.assertFalse(backPropagatedGradient.getOutput().getExampleCount() == 0);
		Assertions.assertFalse(backPropagatedGradient.getOutput().getFeatureCount() == 0);

		Assertions.assertSame(110, backPropagatedGradient.getOutput().getFeatureCount());
		Assertions.assertSame(32, backPropagatedGradient.getOutput().getExampleCount());

	}

}
