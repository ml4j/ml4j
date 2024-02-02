package org.ml4j.nn.components;

import org.ml4j.nn.neurons.Neurons;

/**
 * An example of using a custom NeuralComponent as an alternative to one of the default ml4j NeuralComponents 
 * such as DefaultChainableDirectedComponent.
 * 
 * Such custom components can be created in InceptionV4 graphs by a InceptionV4Definition in the same way as the default components
 * 
 * @author Michael Lavelle
 */
public class ComponentMetadata extends NeuralComponentAdapter<Neurons, Neurons, ComponentMetadata> implements NeuralComponent<ComponentMetadata> {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	private Neurons inputNeurons;
	private Neurons outputNeurons;
	private String description;
	
	public ComponentMetadata(String name, Neurons inputNeurons, Neurons outputNeurons, String description) {
		super(name, NeuralComponentType.createCustomBaseType(ComponentMetadata.class.getName()), inputNeurons, outputNeurons);
		this.description = description;
		this.inputNeurons = inputNeurons;
		this.outputNeurons = outputNeurons;
	}

	public String getDescription() {
		return description;
	}

	@Override
	public String toString() {
		return "ComponentMetadata [inputNeurons=" + inputNeurons + ", outputNeurons=" + outputNeurons + ", description="
				+ description + "]";
	}

	@Override
	public String accept(NeuralComponentVisitor<ComponentMetadata> visitor) {
		return visitor.visitComponent(this);
	}
}
