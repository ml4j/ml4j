/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.architectures.inception.inceptionv4;

import org.ml4j.nn.architectures.inception.inceptionv4.modules.InceptionADefinition;
import org.ml4j.nn.architectures.inception.inceptionv4.modules.InceptionBDefinition;
import org.ml4j.nn.architectures.inception.inceptionv4.modules.InceptionCDefinition;
import org.ml4j.nn.architectures.inception.inceptionv4.modules.InceptionV4StemDefinition;
import org.ml4j.nn.architectures.inception.inceptionv4.modules.InceptionV4TailDefinition;
import org.ml4j.nn.architectures.inception.inceptionv4.modules.ReductionADefinition;
import org.ml4j.nn.architectures.inception.inceptionv4.modules.ReductionBDefinition;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.definitions.Component3DtoNon3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * @author Michael Lavelle
 */
public class InceptionV4Definition implements Component3DtoNon3DGraphDefinition {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private InceptionV4WeightsLoader weightsLoader;
	private float finalDenseLayerRegularisationLambda;
	private float finalDenseLayerInputDropoutKeepProbability;

	public InceptionV4Definition(
			InceptionV4WeightsLoader weightsLoader) {
		this.weightsLoader = weightsLoader;
		this.finalDenseLayerInputDropoutKeepProbability = 1f;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(299, 299, 3, false);
	}

	@Override
	public Neurons getOutputNeurons() {
		return new Neurons(1001, false);
	}

	@Override
	public <T extends NeuralComponent<?>> InitialComponentsGraphBuilder<T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start, NeuralComponentFactory<T> neuralComponentFactory) {
		
		InceptionV4TailDefinition tailDefinition = new InceptionV4TailDefinition(weightsLoader);
		tailDefinition.setDropoutKeepProbability(finalDenseLayerInputDropoutKeepProbability);
		tailDefinition.setRegularisationLambda(finalDenseLayerRegularisationLambda);
		
		return start
				// Initial Stem...
				.withComponentDefinition(new InceptionV4StemDefinition(weightsLoader))
				// followed by 4 InceptionA modules...
				.withComponentDefinition(new InceptionADefinition(weightsLoader, 0))
				.withComponentDefinition(new InceptionADefinition(weightsLoader, 1))
				.withComponentDefinition(new InceptionADefinition(weightsLoader, 2))
				.withComponentDefinition(new InceptionADefinition(weightsLoader, 3))
				// followed by a ReductionA module...
				.withComponentDefinition(new ReductionADefinition(weightsLoader))
				// followed by 7 InceptionB modules...
				.withComponentDefinition(new InceptionBDefinition(weightsLoader, 0))
				.withComponentDefinition(new InceptionBDefinition(weightsLoader, 1))
				.withComponentDefinition(new InceptionBDefinition(weightsLoader, 2))
				.withComponentDefinition(new InceptionBDefinition(weightsLoader, 3))
				.withComponentDefinition(new InceptionBDefinition(weightsLoader, 4))
				.withComponentDefinition(new InceptionBDefinition(weightsLoader, 5))
				.withComponentDefinition(new InceptionBDefinition(weightsLoader, 6))
				// followed by a ReductionB module...
				.withComponentDefinition(new ReductionBDefinition(weightsLoader))
				// followed by 3 InceptionC modules...
				.withComponentDefinition(new InceptionCDefinition(weightsLoader, 0))
				.withComponentDefinition(new InceptionCDefinition(weightsLoader, 1))
				.withComponentDefinition(new InceptionCDefinition(weightsLoader, 2))
				// ending with final Tail
				.withComponentDefinition(tailDefinition);
	}

	public void setFinalDenseLayerRegularisationLambda(float finalDenseLayerRegularisationLambda) {
		this.finalDenseLayerRegularisationLambda = finalDenseLayerRegularisationLambda;
	}

	public void setFinalDenseLayerInputDropoutKeepProbability(float finalDenseLayerInputDropoutKeepProbability) {
		this.finalDenseLayerInputDropoutKeepProbability = finalDenseLayerInputDropoutKeepProbability;
	}

	@Override
	public String getName() {
		return "inception_v4_graph";
	}
}
