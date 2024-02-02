package org.ml4j.nn.neurons.format.features;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;


public class FeaturesFormatTest {
	
	@Test
	public void testDepthHeightWidthFormatDimensionsSize() {
		Assertions.assertTrue(ImageFeaturesFormat.DEPTH_HEIGHT_WIDTH.getDimensions().size() == 3);
	}
	
	@Test
	public void testDepthHeightWidthFormatToString() {
		Assertions.assertEquals("[Depth, Height, Width]", ImageFeaturesFormat.DEPTH_HEIGHT_WIDTH.toString());
	}

	@Test
	public void testDepthHeightWidthFormatEquivalenceInInputScope() {
		Assertions.assertTrue(Dimension.isEquivalent(ImageFeaturesFormat.DEPTH_HEIGHT_WIDTH.getDimensions(),
				Arrays.asList(Dimension.INPUT_DEPTH, Dimension.INPUT_HEIGHT, Dimension.INPUT_WIDTH),
				DimensionScope.INPUT));
	}
	
	@Test
	public void testDepthHeightWidthFormatNonEquivalenceInInputScope() {
		Assertions.assertFalse(Dimension.isEquivalent(ImageFeaturesFormat.DEPTH_HEIGHT_WIDTH.getDimensions(),
				Arrays.asList(Dimension.OUTPUT_DEPTH, Dimension.OUTPUT_HEIGHT, Dimension.OUTPUT_WIDTH),
				DimensionScope.INPUT));
	}
	
	@Test
	public void testDepthHeightWidthFormatNonEquivalenceInOutputScope() {
		Assertions.assertFalse(Dimension.isEquivalent(ImageFeaturesFormat.DEPTH_HEIGHT_WIDTH.getDimensions(),
				Arrays.asList(Dimension.INPUT_DEPTH, Dimension.INPUT_HEIGHT, Dimension.INPUT_WIDTH),
				DimensionScope.OUTPUT));
	}
	
	@Test
	public void testDepthHeightWidthFormatEquivalenceInOutputScope() {
		Assertions.assertTrue(Dimension.isEquivalent(ImageFeaturesFormat.DEPTH_HEIGHT_WIDTH.getDimensions(),
				Arrays.asList(Dimension.OUTPUT_DEPTH, Dimension.OUTPUT_HEIGHT, Dimension.OUTPUT_WIDTH),
				DimensionScope.OUTPUT));
	}
	
	@Test
	public void testDepthHeightWidthInputFormatEquivalenceInAnyScope() {
		Assertions.assertTrue(Dimension.isEquivalent(ImageFeaturesFormat.DEPTH_HEIGHT_WIDTH.getDimensions(),
				Arrays.asList(Dimension.INPUT_DEPTH, Dimension.INPUT_HEIGHT, Dimension.INPUT_WIDTH),
				DimensionScope.ANY));
	}
	
	@Test
	public void testDepthHeightWidthFormatOutputEquivalenceInAnyScope() {
		Assertions.assertTrue(Dimension.isEquivalent(ImageFeaturesFormat.DEPTH_HEIGHT_WIDTH.getDimensions(),
				Arrays.asList(Dimension.OUTPUT_DEPTH, Dimension.OUTPUT_HEIGHT, Dimension.OUTPUT_WIDTH),
				DimensionScope.ANY));
	}
}
