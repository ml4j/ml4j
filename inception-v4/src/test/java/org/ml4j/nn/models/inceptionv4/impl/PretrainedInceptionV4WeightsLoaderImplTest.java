package org.ml4j.nn.models.inceptionv4.impl;

import java.util.Arrays;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.architectures.inception.inceptionv4.InceptionV4WeightsLoader;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

public class PretrainedInceptionV4WeightsLoaderImplTest {

	@Mock
	private MatrixFactory mockMatrixFactory;
	
	@Mock
	private Matrix mockMatrix;

	@BeforeEach
	public void setUp() {
		MockitoAnnotations.initMocks(this);
		Mockito.when(mockMatrixFactory.createMatrixFromRowsByRowsArray(Mockito.eq(32), 
				Mockito.eq(27), Mockito.any())).thenReturn(mockMatrix);
	}

	@Test
	public void testGetConvolutionalLayerWeights() {

		InceptionV4WeightsLoader weightsLoader = new PretrainedInceptionV4WeightsLoaderImpl(
				PretrainedInceptionV4WeightsLoaderImplTest.class.getClassLoader(), mockMatrixFactory);

		WeightsMatrix weightsMatrix = weightsLoader.getConvolutionalLayerWeights("conv2d_1_kernel0", 3, 3, 3, 32);

		Assertions.assertEquals(Arrays.asList(Dimension.OUTPUT_DEPTH, Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT,
				Dimension.FILTER_WIDTH), weightsMatrix.getFormat().getDimensions());

		Assertions.assertEquals(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT,
				Dimension.FILTER_WIDTH), weightsMatrix.getFormat().getInputDimensions());

		Assertions.assertEquals(Arrays.asList(Dimension.OUTPUT_DEPTH), weightsMatrix.getFormat().getOutputDimensions());

		Assertions.assertEquals(WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS, weightsMatrix.getFormat().getOrientation());

		Assertions.assertNotNull(weightsMatrix.getMatrix());

		Assertions.assertEquals(mockMatrix, weightsMatrix.getMatrix());

	}
}
