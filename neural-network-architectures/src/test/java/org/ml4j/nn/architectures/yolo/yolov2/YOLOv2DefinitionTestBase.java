package org.ml4j.nn.architectures.yolo.yolov2;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinitionTestBase;
import org.ml4j.nn.sessions.Session;
import org.ml4j.nn.sessions.SessionImpl;
import org.mockito.Mock;

public abstract class YOLOv2DefinitionTestBase<T extends NeuralComponent<?>> extends Component3Dto3DGraphDefinitionTestBase<T, YOLOv2Definition> {
	
	@Mock
	protected YOLOv2WeightsLoader mockYOLOv2WeightsLoader;
	

	protected abstract void runAssertionsOnCreatedComponentGraph(YOLOv2Definition yOLOv2Definition, 
			InitialComponents3DGraphBuilder<T>  componentGraph);
	
	@Override
	protected YOLOv2Definition createDefinitionToTest() {
		return new YOLOv2Definition(mockYOLOv2WeightsLoader);
	}

	@Override
	protected Session<T> createSession(NeuralComponentFactory<T> componentFactory, DirectedComponentsContext context) {
		return new SessionImpl<>(componentFactory, context);
	}

}
