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
import java.util.Optional;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.base.TestBase;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatchActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
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
public abstract class DefaultDirectedComponentChainBatchTestBase extends TestBase {

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
	protected DirectedAxonsComponent<?, ?, ?> mockComponent1;

	@Mock
	protected DirectedAxonsComponent<?, ?, ?> mockComponent2;
	
	@Mock
	protected AxonsContext mockAxonsContext1;
	
	@Mock
	protected AxonsContext mockAxonsContext2;

	@BeforeEach
	public void setup() {
		MockitoAnnotations.initMocks(this);

		Mockito.when(mockDirectedComponentsContext.getMatrixFactory()).thenReturn(matrixFactory);

		Mockito.when(mockComponent1.isSupported(Mockito.any())).thenReturn(true);
		Mockito.when(mockComponent1.getName()).thenReturn("mockComponent1");
		Mockito.when(mockComponent2.getName()).thenReturn("mockComponent2");
		Mockito.when(mockComponent2.isSupported(Mockito.any())).thenReturn(true);
		Mockito.when(mockComponent1.optimisedFor()).thenReturn(Optional.empty());
		Mockito.when(mockComponent2.optimisedFor()).thenReturn(Optional.empty());
		this.mockNeuronsActivation1 = createNeuronsActivation(100, 32);
		this.mockNeuronsActivation2 = createNeuronsActivation(200, 32);
		this.mockNeuronsActivation3 = createNeuronsActivation(300, 32);
		this.mockNeuronsActivation4 = createNeuronsActivation(400, 32);

		Mockito.when(mockComponent1.getInputNeurons()).thenReturn(new Neurons(100, false));
		Mockito.when(mockComponent2.getInputNeurons()).thenReturn(new Neurons(200, false));
		Mockito.when(mockComponent1.getOutputNeurons()).thenReturn(new Neurons(300, false));
		Mockito.when(mockComponent2.getOutputNeurons()).thenReturn(new Neurons(400, false));
		
		Mockito.when(mockComponent1.getContext(mockDirectedComponentsContext)).thenReturn(mockAxonsContext1);
		Mockito.when(mockComponent2.getContext(mockDirectedComponentsContext)).thenReturn(mockAxonsContext2);

		Mockito.when(mockComponent1.forwardPropagate(Mockito.eq(mockNeuronsActivation1), Mockito.same(mockAxonsContext1)))
				.thenReturn(mockComponent1Activation);
		Mockito.when(mockComponent2.forwardPropagate(Mockito.eq(mockNeuronsActivation2), Mockito.same(mockAxonsContext2)))
				.thenReturn(mockComponent2Activation);
		Mockito.when(mockComponent1Activation.getOutput()).thenReturn(mockNeuronsActivation3);
		Mockito.when(mockComponent2Activation.getOutput()).thenReturn(mockNeuronsActivation4);

	}

	private DefaultDirectedComponentChainBatch createDefaultDirectedComponentChainBatch(
			List<DefaultChainableDirectedComponent<?, ?>> parallelComponents) {
		return createDefaultDirectedComponentChainBatchUnderTest(parallelComponents);
	}

	protected abstract DefaultDirectedComponentChainBatch createDefaultDirectedComponentChainBatchUnderTest(
			List<DefaultChainableDirectedComponent<?, ?>> components);

	@Test
	public void testConstruction() {

		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChainBatch chainBatch = createDefaultDirectedComponentChainBatch(mockComponents);
		Assertions.assertNotNull(chainBatch);
	}

	@Test
	public void testGetComponentType() {
		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChainBatch chainBatch = createDefaultDirectedComponentChainBatch(mockComponents);
		Assertions.assertNotNull(chainBatch);
		Assertions.assertEquals(NeuralComponentBaseType.COMPONENT_CHAIN_BATCH, chainBatch.getComponentType().getBaseType());
	}

	@Test
	public void testForwardPropagate() {

		List<DefaultChainableDirectedComponent<?, ?>> mockComponents = Arrays.asList(mockComponent1, mockComponent2);
		DefaultDirectedComponentChainBatch chainBatch = createDefaultDirectedComponentChainBatch(mockComponents);
		Assertions.assertNotNull(chainBatch);

		DefaultDirectedComponentChainBatchActivation chainBatchActivation = chainBatch.forwardPropagate(
				Arrays.asList(mockNeuronsActivation1, mockNeuronsActivation2), mockDirectedComponentsContext);
		Assertions.assertNotNull(chainBatchActivation);
		List<NeuronsActivation> outputs = chainBatchActivation.getOutput();
		Assertions.assertNotNull(outputs);
		Assertions.assertEquals(2, outputs.size());
		Assertions.assertEquals(mockNeuronsActivation3, outputs.get(0));
		Assertions.assertEquals(mockNeuronsActivation4, outputs.get(1));
	}

	@Test
	public void testDup() {

		// TODO

	}

}
