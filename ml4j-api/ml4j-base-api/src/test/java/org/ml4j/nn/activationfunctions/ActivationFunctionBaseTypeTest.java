package org.ml4j.nn.activationfunctions;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ActivationFunctionBaseTypeTest {

	@Test
	public void testProperties() {
		
		for (ActivationFunctionBaseType baseType :ActivationFunctionBaseType.values()) {
			Assertions.assertNotNull(baseType.getParentType());
			Assertions.assertEquals(baseType, baseType.getBaseType());
			Assertions.assertEquals(baseType, baseType.getParentType());
			Assertions.assertEquals(baseType.name(), baseType.getId());
			Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType." + baseType.getId(), baseType.getQualifiedId());
			Assertions.assertEquals(baseType, baseType.getBaseType());
			Assertions.assertFalse(baseType.isCustomBaseType());
			Assertions.assertTrue(baseType.isStandardBaseType());
			Assertions.assertNotNull(baseType.asActivationFunctionType());
			ActivationFunctionType activationFunctionType = baseType.asActivationFunctionType();
			Assertions.assertEquals(baseType, activationFunctionType.getBaseType());
			Assertions.assertEquals(baseType, activationFunctionType.getParentType());
			Assertions.assertFalse(activationFunctionType.isCustomBaseType());
			Assertions.assertTrue(activationFunctionType.isStandardBaseType());
			Assertions.assertEquals(baseType.getId(),activationFunctionType.getId());
			Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType." + baseType.getId(),activationFunctionType.getQualifiedId());
		}
	}
}
