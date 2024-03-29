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
package org.ml4j.nn.components.onetoone.base;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraphActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/**
 * Abstract base class for unit tests for implementations of
 * ManyToOneDirectedComponent.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class DefaultDirectedComponentChainBipoleGraphTestBase extends TestBase {

	protected NeuronsActivation mockInputActivation;

	protected NeuronsActivation mockNeuronsActivation1;

	protected NeuronsActivation mockNeuronsActivation2;

	protected NeuronsActivation mockNeuronsActivation3;

	protected NeuronsActivation mockNeuronsActivation4;

	@Mock
	protected DirectedComponentsContext mockDirectedComponentsContext;

	@Mock
	private DirectedAxonsComponentActivation mockComponent1Activation;

	@Mock
	protected DirectedAxonsComponentActivation mockComponent2Activation;

	@Mock
	protected DirectedComponentFactory mockDirectedComponentFactory;

	@Mock
	protected DirectedAxonsComponent<?, ?, ?> mockComponent1;

	@Mock
	protected DirectedAxonsComponent<?, ?, ?> mockComponent2;

	@BeforeEach
	public void setup() {
		MockitoAnnotations.initMocks(this);

		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);

		mockInputActivation = createNeuronsActivation(10, 32);
		mockNeuronsActivation1 = createNeuronsActivation(100, 32);
		mockNeuronsActivation2 = createNeuronsActivation(200, 32);
		mockNeuronsActivation3 = createNeuronsActivation(300, 32);
		mockNeuronsActivation4 = createNeuronsActivation(400, 32);

		Mockito.when(mockComponent1.forwardPropagate(Mockito.eq(mockNeuronsActivation1), Mockito.any(AxonsContext.class)))
				.thenReturn(mockComponent1Activation);
		Mockito.when(mockComponent2.forwardPropagate(Mockito.eq(mockNeuronsActivation2), Mockito.any(AxonsContext.class)))
				.thenReturn(mockComponent2Activation);
		Mockito.when(mockComponent1Activation.getOutput()).thenReturn(mockNeuronsActivation3);
		Mockito.when(mockComponent2Activation.getOutput()).thenReturn(mockNeuronsActivation4);

	}

	private DefaultDirectedComponentChainBipoleGraphBase createDefaultDirectedComponentChainBipoleGraph(
			DirectedComponentFactory factory, List<DefaultChainableDirectedComponent<?, ?>> parallelComponents,
			PathCombinationStrategy pathCombinationStrategy) {
		return createDefaultDirectedComponentChainBipoleGraphUnderTest(factory, parallelComponents,
				pathCombinationStrategy);
	}

	protected abstract DefaultDirectedComponentChainBipoleGraphBase createDefaultDirectedComponentChainBipoleGraphUnderTest(
			DirectedComponentFactory factory, List<DefaultChainableDirectedComponent<?, ?>> components,
			PathCombinationStrategy pathCombinationStrategy);

	@Test
	public void testConstruction() {

		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChainBipoleGraphBase graph = createDefaultDirectedComponentChainBipoleGraph(
				mockDirectedComponentFactory, mockComponents, PathCombinationStrategy.ADDITION);
		Assertions.assertNotNull(graph);
	}

	@Test
	public void testGetComponentType() {
		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChainBipoleGraphBase graph = createDefaultDirectedComponentChainBipoleGraph(
				mockDirectedComponentFactory, mockComponents, PathCombinationStrategy.ADDITION);
		Assertions.assertNotNull(graph);
		Assertions.assertEquals(NeuralComponentBaseType.COMPONENT_CHAIN_BIPOLE_GRAPH,
				graph.getComponentType().getBaseType());
	}

	@Test
	public void testForwardPropagate() {
		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChainBipoleGraphBase graph = createDefaultDirectedComponentChainBipoleGraph(
				mockDirectedComponentFactory, mockComponents, PathCombinationStrategy.ADDITION);
		Assertions.assertNotNull(graph);

		DefaultDirectedComponentChainBipoleGraphActivation graphActivation = graph.forwardPropagate(mockInputActivation,
				mockDirectedComponentsContext);
		Assertions.assertNotNull(graphActivation);
		NeuronsActivation output = graphActivation.getOutput();
		Assertions.assertNotNull(output);
		Assertions.assertEquals(mockNeuronsActivation3.getExampleCount(), output.getExampleCount());
		Assertions.assertEquals(mockNeuronsActivation3.getFeatureOrientation(), output.getFeatureOrientation());

	}

	@Test
	public void testDup() {

		// TODO

	}

}
