package org.ml4j.nn.neurons.format.features;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.internal.util.collections.Sets;

public class DimensionTest {
	
	// Test a dimension is always equivalent to itself, whatever the scope
	@Test
	public void testHeightIsEquivalentToHeightWithinInputScope() {
		Assertions.assertTrue(Dimension.HEIGHT.isEquivalent(Dimension.HEIGHT, DimensionScope.INPUT));
	}
	
	@Test
	public void testHeightIsEquivalentToHeightWithinOutputScope() {
		Assertions.assertTrue(Dimension.HEIGHT.isEquivalent(Dimension.HEIGHT, DimensionScope.OUTPUT));
	}
	
	@Test
	public void testHeightIsEquivalentToHeightWithinAnyScope() {
		Assertions.assertTrue(Dimension.HEIGHT.isEquivalent(Dimension.HEIGHT, DimensionScope.ANY));
	}
	
	@Test
	public void testInputHeightIsEquivalentToInputHeightWithinInputScope() {
		Assertions.assertTrue(Dimension.INPUT_HEIGHT.isEquivalent(Dimension.INPUT_HEIGHT, DimensionScope.INPUT));
	}
	
	@Test
	public void testInputHeightIsEquivalentToInputHeightWithinOutputScope() {
		Assertions.assertTrue(Dimension.INPUT_HEIGHT.isEquivalent(Dimension.INPUT_HEIGHT, DimensionScope.OUTPUT));
	}
	
	@Test
	public void testInputHeightIsEquivalentToInputHeightWithinAnyScope() {
		Assertions.assertTrue(Dimension.INPUT_HEIGHT.isEquivalent(Dimension.INPUT_HEIGHT, DimensionScope.ANY));
	}
	
	@Test
	public void testOutputHeightIsEquivalentToOutputHeightWithinInputScope() {
		Assertions.assertTrue(Dimension.OUTPUT_HEIGHT.isEquivalent(Dimension.OUTPUT_HEIGHT, DimensionScope.INPUT));
	}
	
	@Test
	public void testOutputHeightIsEquivalentToOutputHeightWithinOutputScope() {
		Assertions.assertTrue(Dimension.OUTPUT_HEIGHT.isEquivalent(Dimension.OUTPUT_HEIGHT, DimensionScope.OUTPUT));
	}
	
	@Test
	public void testOutputHeightIsEquivalentToOutputHeightWithinAnyScope() {
		Assertions.assertTrue(Dimension.OUTPUT_HEIGHT.isEquivalent(Dimension.OUTPUT_HEIGHT, DimensionScope.ANY));
	}
	
	// Test that a dimension is always equivalent to it's scoped dimension within a scope matching the scoped dimension
	
	@Test
	public void testHeightIsEquivalentToInputHeightWithinInputScope() {
		Assertions.assertTrue(Dimension.HEIGHT.isEquivalent(Dimension.INPUT_HEIGHT, DimensionScope.INPUT));
	}
	
	@Test
	public void testInputHeightIsEquivalentToHeightWithinInputScope() {
		Assertions.assertTrue(Dimension.INPUT_HEIGHT.isEquivalent(Dimension.HEIGHT, DimensionScope.INPUT));
	}
	
	@Test
	public void testHeightIsEquivalentToOutputHeightWithinOutputScope() {
		Assertions.assertTrue(Dimension.HEIGHT.isEquivalent(Dimension.OUTPUT_HEIGHT, DimensionScope.OUTPUT));
	}
	
	@Test
	public void testOutputHeightIsEquivalentToHeightWithinOutputScope() {
		Assertions.assertTrue(Dimension.OUTPUT_HEIGHT.isEquivalent(Dimension.HEIGHT, DimensionScope.OUTPUT));
	}
	
	// Test that a dimension is always not equal to it's scoped dimension within a non-matching scope
	
	@Test
	public void testHeightIsNotEquivalentToInputHeightWithinOutputScope() {
		Assertions.assertFalse(Dimension.HEIGHT.isEquivalent(Dimension.INPUT_HEIGHT, DimensionScope.OUTPUT));
	}
	
	@Test
	public void testInputHeightIsNotEquivalentToHeightWithinOutputScope() {
		Assertions.assertFalse(Dimension.INPUT_HEIGHT.isEquivalent(Dimension.HEIGHT, DimensionScope.OUTPUT));
	}
	
	@Test
	public void testHeightIsNotEquivalentToOutputHeightWithinInputScope() {
		Assertions.assertFalse(Dimension.HEIGHT.isEquivalent(Dimension.OUTPUT_HEIGHT, DimensionScope.INPUT));
	}
	
	@Test
	public void testOutputHeightIsNotEquivalentToHeightWithinInputScope() {
		Assertions.assertFalse(Dimension.OUTPUT_HEIGHT.isEquivalent(Dimension.HEIGHT, DimensionScope.INPUT));
	}
	
	// Test that a dimension is always equal to it's scoped dimension within the ANY scope
	
	@Test
	public void testHeightIsEquivalentToInputHeightWithinAnyScope() {
		Assertions.assertTrue(Dimension.HEIGHT.isEquivalent(Dimension.INPUT_HEIGHT, DimensionScope.ANY));
	}
	
	@Test
	public void testInputHeightIsEquivalentToHeightWithinAnyScope() {
		Assertions.assertTrue(Dimension.INPUT_HEIGHT.isEquivalent(Dimension.HEIGHT, DimensionScope.ANY));
	}
	
	@Test
	public void testHeightIsEquivalentToOutputHeightWithinAnyScope() {
		Assertions.assertTrue(Dimension.HEIGHT.isEquivalent(Dimension.OUTPUT_HEIGHT, DimensionScope.ANY));
	}
	
	@Test
	public void testOutputHeightIsEquivalentToHeightWithinAnyScope() {
		Assertions.assertTrue(Dimension.OUTPUT_HEIGHT.isEquivalent(Dimension.HEIGHT, DimensionScope.ANY));
	}
	
	// Test synonyms
	
	@Test
	public void testDepthIsEqualivalentToChannelWithinInputScope() {
		Assertions.assertTrue(Dimension.DEPTH.isEquivalent(Dimension.CHANNEL, DimensionScope.INPUT));
	}
	
	@Test
	public void testChannelIsEqualivalentToDepthWithinInputScope() {
		Assertions.assertTrue(Dimension.CHANNEL.isEquivalent(Dimension.DEPTH, DimensionScope.INPUT));
	}
	
	@Test
	public void testDepthIsEqualivalentToChannelWithinOutputScope() {
		Assertions.assertTrue(Dimension.DEPTH.isEquivalent(Dimension.CHANNEL, DimensionScope.OUTPUT));
	}
	
	@Test
	public void testChannelIsEqualivalentToDepthWithinOutputScope() {
		Assertions.assertTrue(Dimension.CHANNEL.isEquivalent(Dimension.DEPTH, DimensionScope.OUTPUT));
	}
	
	@Test
	public void testDepthIsEqualivalentToChannelWithinAnyScope() {
		Assertions.assertTrue(Dimension.DEPTH.isEquivalent(Dimension.CHANNEL, DimensionScope.ANY));
	}
	
	@Test
	public void testChannelIsEqualivalentToDepthWithinAnyScope() {
		Assertions.assertTrue(Dimension.CHANNEL.isEquivalent(Dimension.DEPTH, DimensionScope.ANY));
	}
	
	@Test
	public void testGetAliasesOfDepthInAnyScope() {
		Assertions.assertEquals(Sets.newSet(Dimension.CHANNEL), 
				Dimension.DEPTH.getAliases(DimensionScope.ANY));
	}
	
	@Test
	public void testGetAliasesOfDepthInInputScope() {
		Assertions.assertEquals(Sets.newSet(Dimension.INPUT_DEPTH, Dimension.CHANNEL), 
				Dimension.DEPTH.getAliases(DimensionScope.INPUT));
	}
	
	@Test
	public void testGetAliasesOfOutputHeightInInputScope() {
		Assertions.assertTrue(Dimension.OUTPUT_HEIGHT.getAliases(DimensionScope.INPUT).isEmpty());
	}
	
	@Test
	public void testGetAliasesOfOutputHeightInOutputScope() {
		Assertions.assertEquals(Sets.newSet(Dimension.HEIGHT), 
				Dimension.OUTPUT_HEIGHT.getAliases(DimensionScope.OUTPUT));
	}
	
	@Test
	public void testGetAliasesOfOutputHeightInAnyScope() {
		Assertions.assertEquals(Sets.newSet(Dimension.HEIGHT), 
				Dimension.OUTPUT_HEIGHT.getAliases(DimensionScope.ANY));
	}
	
	@Test
	public void testGetAliasesOfDepthInOutputScope() {
		Assertions.assertEquals(Sets.newSet(Dimension.OUTPUT_DEPTH, Dimension.CHANNEL), 
				Dimension.DEPTH.getAliases(DimensionScope.OUTPUT));
	}
	
}
