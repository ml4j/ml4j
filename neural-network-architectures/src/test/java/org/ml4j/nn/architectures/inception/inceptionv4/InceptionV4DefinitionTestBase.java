package org.ml4j.nn.architectures.inception.inceptionv4;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.definitions.Component3DtoNon3DGraphDefinitionTestBase;
import org.ml4j.nn.sessions.Session;
import org.ml4j.nn.sessions.SessionImpl;
import org.mockito.Mock;

public abstract class InceptionV4DefinitionTestBase<T extends NeuralComponent<?>> extends Component3DtoNon3DGraphDefinitionTestBase<T, InceptionV4Definition> {
	
	@Mock
	protected InceptionV4WeightsLoader mockInceptionV4WeightsLoader;
	
	protected abstract void runAssertionsOnCreatedComponentGraph(InceptionV4Definition inceptionV4Definition, 
			InitialComponentsGraphBuilder<T>  componentGraph);

	@Test
	public void testComponentGraphCreation() {
		
		
	
		// Start new session, given the component factory and the runtime context.
		Session<T> session = new SessionImpl<>(neuralComponentFactory, mockDirectedComponentsContext);
		
		// Create the InceptionV4Definition
		InceptionV4Definition inceptionV4Definition = new InceptionV4Definition( 
				mockInceptionV4WeightsLoader);
		
		// Build a component graph, given this Session and the InceptionV4Definition.
		InitialComponentsGraphBuilder<T> componentGraph = session.buildComponentGraph().startWith(inceptionV4Definition);
			
		// Assert that we now have a component graph.
		Assertions.assertNotNull(componentGraph);
		
		// Run additional assertions
		runAssertionsOnCreatedComponentGraph(inceptionV4Definition, componentGraph);
	}

	@Override
	protected InceptionV4Definition createDefinitionToTest() {
		return new InceptionV4Definition(mockInceptionV4WeightsLoader);
	}

	@Override
	protected Session<T> createSession(NeuralComponentFactory<T> componentFactory, DirectedComponentsContext context) {
		return new SessionImpl<>(componentFactory, context);
	}

}
