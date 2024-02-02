package org.ml4j.nn.components;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class NeuralComponentBaseTypeTest {

	@Test
	public void testProperties() {
		
		for (NeuralComponentBaseType baseType :NeuralComponentBaseType.values()) {
			Assertions.assertNotNull(baseType.getParentType());
			Assertions.assertEquals(baseType, baseType.getBaseType());
			Assertions.assertEquals(baseType, baseType.getParentType());
			Assertions.assertEquals(baseType.name(), baseType.getId());
			Assertions.assertEquals(baseType.name(), baseType.getQualifiedId());
			Assertions.assertEquals(baseType, baseType.getBaseType());
			Assertions.assertFalse(baseType.isCustomBaseType());
			Assertions.assertTrue(baseType.isStandardBaseType());
			Assertions.assertNotNull(baseType.asNeuralNetworkType());
			NeuralComponentType neuralComponentType = baseType.asNeuralNetworkType();
			Assertions.assertEquals(baseType, neuralComponentType.getBaseType());
			Assertions.assertEquals(baseType, neuralComponentType.getParentType());
			Assertions.assertFalse(neuralComponentType.isCustomBaseType());
			Assertions.assertTrue(neuralComponentType.isStandardBaseType());
			Assertions.assertEquals(baseType.getId(),neuralComponentType.getId());
			Assertions.assertEquals(baseType.getQualifiedId(),neuralComponentType.getQualifiedId());
		}
	}
}
