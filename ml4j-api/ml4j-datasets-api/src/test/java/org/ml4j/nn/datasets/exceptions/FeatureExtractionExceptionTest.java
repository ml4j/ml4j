package org.ml4j.nn.datasets.exceptions;


import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class FeatureExtractionExceptionTest {

	@Test
	public void testConstructionWithMessage() {
		Assertions.assertNotNull(new FeatureExtractionException("some message"));
	}
	
	@Test
	public void testConstructionWithMessageAndThrowable() {
		Assertions.assertNotNull(new FeatureExtractionException("some message", new RuntimeException("some exception")));
	}
}
