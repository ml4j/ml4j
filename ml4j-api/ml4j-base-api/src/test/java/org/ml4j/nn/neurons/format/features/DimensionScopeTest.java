package org.ml4j.nn.neurons.format.features;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class DimensionScopeTest {

	@Test
	public void testIsInputScopeValidWithinAnyScope() {
		Assertions.assertFalse(DimensionScope.INPUT.isValidWithin(DimensionScope.ANY));
	}
	
	@Test
	public void testIsOutputScopeValidWithinAnyScope() {
		Assertions.assertFalse(DimensionScope.OUTPUT.isValidWithin(DimensionScope.ANY));
	}
	
	@Test
	public void testIsAnyScopeValidWithinAnyScope() {
		Assertions.assertTrue(DimensionScope.ANY.isValidWithin(DimensionScope.ANY));
	}
	
	@Test
	public void testIsInputScopeValidWithinInputScope() {
		Assertions.assertTrue(DimensionScope.INPUT.isValidWithin(DimensionScope.INPUT));
	}
	
	@Test
	public void testIsOutputScopeValidWithinInputScope() {
		Assertions.assertFalse(DimensionScope.OUTPUT.isValidWithin(DimensionScope.INPUT));
	}
	
	@Test
	public void testIsAnyScopeValidWithinInputScope() {
		Assertions.assertTrue(DimensionScope.ANY.isValidWithin(DimensionScope.INPUT));
	}
	
	@Test
	public void testIsInputScopeValidWithinOutputScope() {
		Assertions.assertFalse(DimensionScope.INPUT.isValidWithin(DimensionScope.OUTPUT));
	}
	
	@Test
	public void testIsOutputScopeValidWithinOutputScope() {
		Assertions.assertTrue(DimensionScope.OUTPUT.isValidWithin(DimensionScope.OUTPUT));
	}
	
	@Test
	public void testIsAnyScopeValidWithinOutputScope() {
		Assertions.assertTrue(DimensionScope.ANY.isValidWithin(DimensionScope.OUTPUT));
	}

}
