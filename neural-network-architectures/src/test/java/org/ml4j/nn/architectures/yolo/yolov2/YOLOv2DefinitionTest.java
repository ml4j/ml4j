package org.ml4j.nn.architectures.yolo.yolov2;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.factories.DummyDirectedComponentFactoryImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Integration test with some mock components and some real components, which creates an actual YOLOv2Network,
 * and forward propagates a neurons activation through the network to ensure the output matches the expected output
 * dimensions.
 * 
 * @author Michael Lavelle
 */
public class YOLOv2DefinitionTest extends YOLOv2DefinitionTestBase<DefaultChainableDirectedComponent<?, ?>> {

	private static final Logger LOGGER = LoggerFactory.getLogger(YOLOv2DefinitionTest.class);
	
	private DirectedComponentFactory dummyComponentFactory;
	private MatrixFactory matrixFactory;
	
	public YOLOv2DefinitionTest() {
		this.dummyComponentFactory = new DummyDirectedComponentFactoryImpl();
		this.matrixFactory = new JBlasRowMajorMatrixFactory();
	}
	
	@Override
	protected NeuralComponentFactory<DefaultChainableDirectedComponent<?, ?>> createNeuralComponentFactory() {
		return dummyComponentFactory;
	}

	@Override
	protected void runAssertionsOnCreatedComponentGraph(YOLOv2Definition yoloV2Definition, 
			InitialComponents3DGraphBuilder<DefaultChainableDirectedComponent<?, ?>> componentGraph) {
		
		List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents = componentGraph.getComponents();

		Assertions.assertEquals(49, sequentialComponents.size());
	
		int index = 0;
	
		for (DefaultChainableDirectedComponent<?, ?> component : sequentialComponents) {
			LOGGER.debug("Component:" + index++ + ":" + component.getComponentType() + ": Input Neurons: " + component.getInputNeurons());
		}
		
		// Create a component chain from the components
		DefaultDirectedComponentChain componentChain = dummyComponentFactory.createDirectedComponentChain(sequentialComponents);
	
		int batchSize = 10;
		int inputFeatureCount = yoloV2Definition.getInputNeurons().getNeuronCountExcludingBias();
		int expectedOutputFeatureCount = yoloV2Definition.getOutputNeurons().getNeuronCountExcludingBias();
		
		Matrix inputMatrix = new JBlasRowMajorMatrixFactory().createMatrix(inputFeatureCount, batchSize);

		NeuronsActivation input = new NeuronsActivationImpl(yoloV2Definition.getInputNeurons(), inputMatrix, NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
	
		DefaultChainableDirectedComponentActivation output = componentChain.forwardPropagate(input, mockDirectedComponentsContext);

		Assertions.assertNotNull(output);

		Assertions.assertNotNull(output.getOutput());

		Matrix outputMatrix = output.getOutput().getActivations(matrixFactory);

		Assertions.assertNotNull(outputMatrix);


		Assertions.assertEquals(batchSize, outputMatrix.getColumns());
		Assertions.assertEquals(expectedOutputFeatureCount, outputMatrix.getRows());

		
	}
}
