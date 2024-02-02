package org.ml4j.nn.activationfunctions;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ActivationFunctionTypeTest {
	
	@Test
	public void testGetBaseType() {
		ActivationFunctionBaseType activationFunctionBaseType = ActivationFunctionBaseType.RELU;
		ActivationFunctionType activationFunctionType = ActivationFunctionType.getBaseType(activationFunctionBaseType);
		Assertions.assertNotNull(activationFunctionType);
		Assertions.assertEquals(activationFunctionBaseType, activationFunctionType.getBaseType());
		Assertions.assertEquals(activationFunctionBaseType.getId(), activationFunctionType.getId());
		Assertions.assertEquals(activationFunctionBaseType.getQualifiedId(), activationFunctionType.getQualifiedId());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU", activationFunctionType.getQualifiedId());
		Assertions.assertEquals(activationFunctionBaseType.getParentType(), activationFunctionType.getParentType());
		Assertions.assertEquals(activationFunctionBaseType.getBaseType(), activationFunctionType.getBaseType());
		Assertions.assertEquals(activationFunctionBaseType, activationFunctionType.getBaseType());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU", activationFunctionType.toString());
	}
	
	@Test
	public void testGetBaseTypeWhenBaseTypeIsCustom() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {

			ActivationFunctionBaseType activationFunctionBaseType = ActivationFunctionBaseType.CUSTOM;
			ActivationFunctionType.getBaseType(activationFunctionBaseType);

		});
	}
	
	@Test
	public void testCreateCustomBaseType() {
		ActivationFunctionType activationFunctionType = ActivationFunctionType.createCustomBaseType("SOME_ID");
		Assertions.assertNotNull(activationFunctionType);
		Assertions.assertEquals(ActivationFunctionBaseType.CUSTOM, activationFunctionType.getBaseType());
		Assertions.assertEquals("SOME_ID", activationFunctionType.getId());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.SOME_ID", activationFunctionType.getQualifiedId());		
		Assertions.assertEquals(ActivationFunctionBaseType.CUSTOM.getBaseType(), activationFunctionType.getParentType());	
		Assertions.assertEquals(ActivationFunctionBaseType.CUSTOM.asActivationFunctionType().getId(), activationFunctionType.getParentType().getId());	
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.SOME_ID", activationFunctionType.toString());
	}
	
	@Test
	public void testCreateSubType() {
		ActivationFunctionType activationFunctionSubType = ActivationFunctionType.createSubType(ActivationFunctionBaseType.RELU, "SOME_ID");
		Assertions.assertNotNull(activationFunctionSubType);
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionSubType.getBaseType());
		Assertions.assertEquals("SOME_ID", activationFunctionSubType.getId());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID", activationFunctionSubType.getQualifiedId());
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionSubType.getParentType());	
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID", activationFunctionSubType.toString());

	}
	
	@Test
	public void testCreateSubTypeWhenParentIsActivationFunctionType() {
		ActivationFunctionType activationFunctionSubType = ActivationFunctionType.createSubType(ActivationFunctionBaseType.RELU.asActivationFunctionType(), "SOME_ID");
		Assertions.assertNotNull(activationFunctionSubType);
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionSubType.getBaseType());
		Assertions.assertEquals("SOME_ID", activationFunctionSubType.getId());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID", activationFunctionSubType.getQualifiedId());
		Assertions.assertEquals(ActivationFunctionBaseType.RELU.asActivationFunctionType(), activationFunctionSubType.getParentType());	
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID", activationFunctionSubType.toString());
	}
	
	@Test
	public void testCreateSubTypeWhenParentIsSubtypeActivationFunctionType() {
		ActivationFunctionType parentType = ActivationFunctionType.createSubType(ActivationFunctionBaseType.RELU.asActivationFunctionType(), "SOME_ID_1");
		ActivationFunctionType activationFunctionSubType = ActivationFunctionType.createSubType(parentType, "SOME_ID_2");
		Assertions.assertNotNull(activationFunctionSubType);
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionSubType.getBaseType());
		Assertions.assertEquals("SOME_ID_2", activationFunctionSubType.getId());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID_1.SOME_ID_2", activationFunctionSubType.getQualifiedId());
		Assertions.assertEquals(parentType, activationFunctionSubType.getParentType());		
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID_1.SOME_ID_2", activationFunctionSubType.toString());

	}
	
	@Test
	public void testCreateCustomBaseTypeNameClash() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {

			ActivationFunctionType.createCustomBaseType("RELU");

		});
	}
	
	@Test
	public void testConstructorWhenNotBaseType() {
		ActivationFunctionType activationFunctionType = new ActivationFunctionType(ActivationFunctionBaseType.RELU, "SOME_ID", false, false);
		Assertions.assertEquals("SOME_ID", activationFunctionType.getId());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID", activationFunctionType.getQualifiedId());
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionType.getBaseType());
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionType.getParentType());
		Assertions.assertFalse(activationFunctionType.isStandardBaseType());
		Assertions.assertFalse(activationFunctionType.isCustomBaseType());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID", activationFunctionType.toString());

	}
	
	@Test
	public void testConstructorWhenStandardBaseTypeAndIdDoesNotMatch() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {

			new ActivationFunctionType(ActivationFunctionBaseType.RELU, "SOME_ID", true, false);

		});
	}
	
	@Test
	public void testConstructorWhenStandardBaseTypeAndIdMatches() {
		ActivationFunctionType activationFunctionType = new ActivationFunctionType(ActivationFunctionBaseType.RELU, "RELU", true, false);
		Assertions.assertEquals("RELU", activationFunctionType.getId());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU", activationFunctionType.getQualifiedId());
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionType.getBaseType());
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionType.getParentType());
		Assertions.assertTrue(activationFunctionType.isStandardBaseType());
		Assertions.assertFalse(activationFunctionType.isCustomBaseType());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU", activationFunctionType.toString());

	}
	
	@Test
	public void testConstructorWhenStandardBaseTypeAndIdMatchesAndParentTypeIsAnActivationFunctionTypeInstance() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {

			new ActivationFunctionType(ActivationFunctionBaseType.RELU.asActivationFunctionType(), "RELU", true, false);

		});
	}
	
	@Test
	public void testConstructorWhenCustomBaseType() {
		ActivationFunctionType activationFunctionType = new ActivationFunctionType(ActivationFunctionBaseType.RELU, "SOME_ID", false, true);
		Assertions.assertEquals("SOME_ID", activationFunctionType.getId());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID", activationFunctionType.getQualifiedId());
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionType.getBaseType());
		Assertions.assertEquals(ActivationFunctionBaseType.RELU, activationFunctionType.getParentType());
		Assertions.assertFalse(activationFunctionType.isStandardBaseType());
		Assertions.assertTrue(activationFunctionType.isCustomBaseType());
		Assertions.assertEquals("org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.RELU.SOME_ID", activationFunctionType.toString());
	}
	
	@Test
	public void testHashCode() {
		ActivationFunctionType activationFunctionType1 = new ActivationFunctionType(ActivationFunctionBaseType.RELU, "RELU", true, false);
		ActivationFunctionType activationFunctionType2 = new ActivationFunctionType(ActivationFunctionBaseType.RELU, "RELU", true, false);
		Assertions.assertEquals(activationFunctionType1.hashCode(), activationFunctionType2.hashCode());
	}
	

	@Test
	public void testConstructorWhenCustomBaseTypeAndStandardBaseTypeAndNameMatches() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {

			new ActivationFunctionType(ActivationFunctionBaseType.RELU, "RELU", true, true);

		});
	}

}
