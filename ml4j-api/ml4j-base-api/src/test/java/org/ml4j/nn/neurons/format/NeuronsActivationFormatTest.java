package org.ml4j.nn.neurons.format;

import java.util.Arrays;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.ml4j.nn.neurons.format.features.FeaturesFormat;
import org.ml4j.nn.neurons.format.features.FeaturesFormatImpl;

public class NeuronsActivationFormatTest {

	@Test
	public void testRowsSpanFeatureSetFormatRowDimensions() {
		Assertions.assertEquals(Arrays.asList(Dimension.FEATURE),
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET.getRowDimensions());	
	}
	
	@Test
	public void testRowsSpanFeatureSetFormatRowDimensionsName() {
		Assertions.assertEquals("[Feature]", 
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET.getRowDimensionsName());	
	}
	
	@Test
	public void testRowsSpanFeatureSetFormatColumnDimensionsName() {
		Assertions.assertEquals("[Example]", 
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET.getColumnDimensionsName());	
	}
	
	@Test
	public void testRowsSpanFeatureSetFormatColumnDimensions() {
		Assertions.assertEquals(Arrays.asList(Dimension.EXAMPLE), 
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET.getColumnDimensions());	
	}
	
	@Test
	public void testRowsSpanFeatureSetFormatDimensions() {
		Assertions.assertEquals(Arrays.asList(Dimension.FEATURE, Dimension.EXAMPLE), 
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET.getDimensions());	
	}
	
	@Test
	public void testRowsSpanFeatureSetExampleDimensions() {
		Assertions.assertEquals(Arrays.asList(Dimension.EXAMPLE), 
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET.getExampleDimensions());	
	}
	
	@Test
	public void testRowsSpanFeatureSetGetFeatureFormat() {
		Assertions.assertEquals(FeaturesFormat.FLAT, 
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET.getFeaturesFormat());	
	}
	
	@Test
	public void testRowsSpanFeatureSetGetFeatureOrientation() {
		Assertions.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, 
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET.getFeatureOrientation());	
	}
	
	@Test
	public void testColumnsSpanFeatureSetFormatRowDimensions() {
		Assertions.assertEquals(Arrays.asList(Dimension.EXAMPLE), 
				NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET.getRowDimensions());	
	}
	
	@Test
	public void testColumnsSpanFeatureSetFormatRowDimensionsName() {
		Assertions.assertEquals("[Example]", 
				NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET.getRowDimensionsName());	
	}
	
	@Test
	public void testColumnsSpanFeatureSetFormatColumnDimensionsName() {
		Assertions.assertEquals("[Feature]", 
				NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET.getColumnDimensionsName());	
	}
	
	@Test
	public void testColumnsSpanFeatureSetFormatColumnDimensions() {
		Assertions.assertEquals(Arrays.asList(Dimension.FEATURE), 
				NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET.getColumnDimensions());	
	}
	
	@Test
	public void testColumnsSpanFeatureSetFormatDimensions() {
		Assertions.assertEquals(Arrays.asList(Dimension.EXAMPLE, Dimension.FEATURE), 
				NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET.getDimensions());	
	}
	
	@Test
	public void testColumnsSpanFeatureSetExampleDimensions() {
		Assertions.assertEquals(Arrays.asList(Dimension.EXAMPLE), 
				NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET.getExampleDimensions());	
	}
	
	@Test
	public void testColumnsSpanFeatureSetGetFeatureFormat() {
		Assertions.assertEquals(FeaturesFormat.FLAT, 
				NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET.getFeaturesFormat());	
	}
	
	@Test
	public void testColumnsSpanFeatureSetGetFeatureOrientation() {
		Assertions.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, 
				NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET.getFeatureOrientation());	
	}
	
	@Test
	public void testCustomNeuronsActivationFormat() {
		
		NeuronsActivationFormat<?> customDepthHeightWidthWithExampleColumnsFormat = new NeuronsActivationFormat<>(
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, new FeaturesFormatImpl(
				Arrays.asList(Dimension.DEPTH, Dimension.HEIGHT, Dimension.WIDTH)), 
				Arrays.asList(Dimension.EXAMPLE));

		
		Assertions.assertTrue(ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT
				.isEquivalentFormat(customDepthHeightWidthWithExampleColumnsFormat, DimensionScope.ANY));
		
		Assertions.assertTrue(ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT
				.isEquivalentFormat(customDepthHeightWidthWithExampleColumnsFormat, DimensionScope.INPUT));
		
		Assertions.assertTrue(ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT
				.isEquivalentFormat(customDepthHeightWidthWithExampleColumnsFormat, DimensionScope.OUTPUT));
		
		Assertions.assertTrue(Dimension.isEquivalent(
				customDepthHeightWidthWithExampleColumnsFormat.getDimensions(), ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT.getDimensions(), DimensionScope.ANY));
		
		Assertions.assertTrue(Dimension.isEquivalent(
				customDepthHeightWidthWithExampleColumnsFormat.getDimensions(), ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT.getDimensions(), DimensionScope.INPUT));
	
		Assertions.assertTrue(Dimension.isEquivalent(
				customDepthHeightWidthWithExampleColumnsFormat.getDimensions(), ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT.getDimensions(), DimensionScope.OUTPUT));
		
	}
	
	@Test
	public void testCustomNeuronsActivationFormat2() {
		
		NeuronsActivationFormat<?> customDepthHeightWidthWithExampleColumnsFormat = new NeuronsActivationFormat<>(
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, new FeaturesFormatImpl(
				Arrays.asList(Dimension.INPUT_DEPTH, Dimension.INPUT_HEIGHT, Dimension.INPUT_WIDTH)), 
				Arrays.asList(Dimension.EXAMPLE));
		
		Assertions.assertTrue(ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT
				.isEquivalentFormat(customDepthHeightWidthWithExampleColumnsFormat, DimensionScope.ANY));
		
		Assertions.assertTrue(ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT
				.isEquivalentFormat(customDepthHeightWidthWithExampleColumnsFormat, DimensionScope.INPUT));
		
		Assertions.assertFalse(ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT
				.isEquivalentFormat(customDepthHeightWidthWithExampleColumnsFormat, DimensionScope.OUTPUT));
		
		Assertions.assertTrue(Dimension.isEquivalent(
				customDepthHeightWidthWithExampleColumnsFormat.getDimensions(), ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT.getDimensions(), DimensionScope.ANY));
		
		Assertions.assertTrue(Dimension.isEquivalent(
				customDepthHeightWidthWithExampleColumnsFormat.getDimensions(), ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT.getDimensions(), DimensionScope.INPUT));
	
		Assertions.assertFalse(Dimension.isEquivalent(
				customDepthHeightWidthWithExampleColumnsFormat.getDimensions(), ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT.getDimensions(), DimensionScope.OUTPUT));
		
	}
}
