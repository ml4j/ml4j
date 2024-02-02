package org.ml4j.nn.axons;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class AxonsBaseTypeTest {

	@Test
	public void testProperties() {
		
		for (AxonsBaseType baseType :AxonsBaseType.values()) {
			Assertions.assertNotNull(baseType.getParentType());
			Assertions.assertEquals(baseType, baseType.getBaseType());
			Assertions.assertEquals(baseType, baseType.getParentType());
			Assertions.assertEquals(baseType.name(), baseType.getId());
			Assertions.assertEquals(baseType.getQualifiedId(), baseType.getQualifiedId());
			Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType." + baseType.getId(), baseType.getQualifiedId());
			Assertions.assertEquals(baseType, baseType.getBaseType());
			Assertions.assertFalse(baseType.isCustomBaseType());
			Assertions.assertTrue(baseType.isStandardBaseType());
			Assertions.assertNotNull(baseType.asAxonsType());
			AxonsType axonsType = baseType.asAxonsType();
			Assertions.assertEquals(baseType, axonsType.getBaseType());
			Assertions.assertEquals(baseType, axonsType.getParentType());
			Assertions.assertFalse(axonsType.isCustomBaseType());
			Assertions.assertTrue(axonsType.isStandardBaseType());
			Assertions.assertEquals(baseType.getId(),axonsType.getId());
			Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType." + baseType.getId(),axonsType.getQualifiedId());
		}
	}
}
