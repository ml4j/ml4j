package org.ml4j.nn.datasets;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;


public class FeatureExtractionErrorModeTest {

	@Test
	public void testValues() {
		Assertions.assertEquals(Arrays.asList(FeatureExtractionErrorMode.IGNORE, FeatureExtractionErrorMode.LOG_WARNING, FeatureExtractionErrorMode.RAISE_EXCEPTION), Arrays.asList(FeatureExtractionErrorMode.values()));
	}
}
