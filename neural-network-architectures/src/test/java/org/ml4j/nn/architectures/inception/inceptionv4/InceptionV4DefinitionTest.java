package org.ml4j.nn.architectures.inception.inceptionv4;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
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
 * Integration test with some mock components and some real components, which creates an actual InceptionV4Network,
 * and forward propagates a neurons activation through the network to ensure the output matches the expected output
 * dimensions.
 * 
 * @author Michael Lavelle
 */
public class InceptionV4DefinitionTest extends InceptionV4DefinitionTestBase<DefaultChainableDirectedComponent<?, ?>> {

	private static final Logger LOGGER = LoggerFactory.getLogger(InceptionV4DefinitionTest.class);
	
	private DirectedComponentFactory dummyComponentFactory;
	private MatrixFactory matrixFactory;
	
	public InceptionV4DefinitionTest() {
		this.dummyComponentFactory = new DummyDirectedComponentFactoryImpl();
		this.matrixFactory = new JBlasRowMajorMatrixFactory();
	}
	
	@Override
	protected NeuralComponentFactory<DefaultChainableDirectedComponent<?, ?>> createNeuralComponentFactory() {
		return dummyComponentFactory;
	}

	@Override
	protected void runAssertionsOnCreatedComponentGraph(InceptionV4Definition inceptionV4Definition, 
			InitialComponentsGraphBuilder<DefaultChainableDirectedComponent<?, ?>> componentGraph) {
		
		List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents = componentGraph.getComponents();
		
		Assertions.assertEquals(31, sequentialComponents.size());
		
		int index = 0;
		for (DefaultChainableDirectedComponent<?, ?> component : sequentialComponents) {
			LOGGER.debug("Component:" + index++ + ":" + component.getComponentType() + ": Input Neurons: " + component.getInputNeurons());
		}
		
		// Create a component chain from the components
		DefaultDirectedComponentChain componentChain = dummyComponentFactory.createDirectedComponentChain(sequentialComponents);
	
		int batchSize = 10;
		int inputFeatureCount = inceptionV4Definition.getInputNeurons().getNeuronCountExcludingBias();
		int expectedOutputFeatureCount = 1001;
		
		Matrix inputMatrix = new JBlasRowMajorMatrixFactory().createMatrix(inputFeatureCount, batchSize);

		NeuronsActivation input = new NeuronsActivationImpl(inceptionV4Definition.getInputNeurons(), inputMatrix, NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
	
		DefaultChainableDirectedComponentActivation output = componentChain.forwardPropagate(input, mockDirectedComponentsContext);
		
		Assertions.assertNotNull(output);

		Assertions.assertNotNull(output.getOutput());

		Matrix outputMatrix = output.getOutput().getActivations(matrixFactory);

		Assertions.assertNotNull(outputMatrix);


		Assertions.assertEquals(batchSize, outputMatrix.getColumns());
		Assertions.assertEquals(expectedOutputFeatureCount, outputMatrix.getRows());

		
	}
}
