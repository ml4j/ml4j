package org.ml4j.nn.components;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

public class DummyGenericComponent
		implements DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, Serializable> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private Neurons inputNeurons;
	private Neurons outputNeurons;
	private NeuralComponentType neuralComponentType;
	private String name;

	public DummyGenericComponent(String name, Neurons inputNeurons, Neurons outputNeurons,
			NeuralComponentType neuralComponentType2) {
		super();
		this.inputNeurons = inputNeurons;
		this.outputNeurons = outputNeurons;
		this.neuralComponentType = neuralComponentType2;
		this.name = name;
	}

	@Override
	public Serializable getContext(DirectedComponentsContext directedComponentsContext) {
		return new Serializable() {

			/**
			 * Default serialization id.
			 */
			private static final long serialVersionUID = 1L;
			
		};
	}

	@Override
	public DefaultChainableDirectedComponentActivation forwardPropagate(NeuronsActivation input, Serializable context) {
		NeuronsActivation dummyOutput = new DummyNeuronsActivation(outputNeurons, input.getFeatureOrientation(),
				input.getExampleCount());
		if (dummyOutput.getFeatureCount() != getOutputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException();
		}
		return new DummyGenericComponentActivation(input, dummyOutput);
	}

	@Override
	public DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, Serializable> dup(DirectedComponentFactory directedComponentFactory) {
		return new DummyGenericComponent(name, inputNeurons, outputNeurons, neuralComponentType);
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public Neurons getInputNeurons() {
		return inputNeurons;
	}

	@Override
	public Neurons getOutputNeurons() {
		return outputNeurons;
	}

	@Override
	public NeuralComponentType getComponentType() {
		return neuralComponentType;
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}

	@Override
	public DefaultChainableDirectedComponentActivation forwardPropagateChain(NeuronsActivation input,
			DirectedComponentsContext context) {
		return forwardPropagate(input, getContext(context));
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public String accept(NeuralComponentVisitor<DefaultChainableDirectedComponent<?, ?>> visitor) {
		return visitor.visitComponent(this);
	}
	
	@Override
	public Set<DefaultChainableDirectedComponent<?, ?>> flatten() {
		Set<DefaultChainableDirectedComponent<?, ?>> allComponentsIncludingThis = new HashSet<>(Arrays.asList(this));
		return allComponentsIncludingThis;
	}
}
