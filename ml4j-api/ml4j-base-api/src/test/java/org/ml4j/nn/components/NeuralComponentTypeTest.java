package org.ml4j.nn.components;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class NeuralComponentTypeTest {

	@Test
	public void testProperties() {
		
		for (NeuralComponentBaseType baseType :NeuralComponentBaseType.values()) {
			Assertions.assertNotNull(baseType.getParentType());
			Assertions.assertEquals(baseType, baseType.getBaseType());
			Assertions.assertEquals(baseType, baseType.getParentType());
			Assertions.assertEquals(baseType.name(), baseType.getId());
			Assertions.assertEquals(baseType.name(), baseType.getQualifiedId());
			Assertions.assertEquals(baseType, baseType.getBaseType());
		}
	}
}
