package org.ml4j.nn.datasets.exceptions;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class FeatureExtractionRuntimeExceptionTest {

	@Test
	public void testConstructionWithMessageAndException() {
		Assertions.assertNotNull(new FeatureExtractionRuntimeException("some message", new RuntimeException("some exception")));
	}
}
