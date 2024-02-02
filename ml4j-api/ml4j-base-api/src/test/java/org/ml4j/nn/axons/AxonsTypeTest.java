package org.ml4j.nn.axons;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class AxonsTypeTest {
	
	@Test
	public void testGetBaseType() {
		AxonsBaseType axonsBaseType = AxonsBaseType.CONVOLUTIONAL;
		AxonsType axonsType = AxonsType.getBaseType(axonsBaseType);
		Assertions.assertNotNull(axonsType);
		Assertions.assertEquals(axonsBaseType, axonsType.getBaseType());
		Assertions.assertEquals(axonsBaseType.getId(), axonsType.getId());
		Assertions.assertEquals(axonsBaseType.getQualifiedId(), axonsType.getQualifiedId());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL", axonsType.getQualifiedId());
		Assertions.assertEquals(axonsBaseType.getParentType(), axonsType.getParentType());
		Assertions.assertEquals(axonsBaseType.getBaseType(), axonsType.getBaseType());
		Assertions.assertEquals(axonsBaseType, axonsType.getBaseType());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL", axonsType.toString());
	}
	
	@Test
	public void testGetBaseTypeWhenBaseTypeIsCustom() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {

			AxonsBaseType axonsBaseType = AxonsBaseType.CUSTOM;
			AxonsType.getBaseType(axonsBaseType);

		});
	}
	
	@Test
	public void testCreateCustomBaseType() {
		AxonsType axonsType = AxonsType.createCustomBaseType("SOME_ID");
		Assertions.assertNotNull(axonsType);
		Assertions.assertEquals(AxonsBaseType.CUSTOM, axonsType.getBaseType());
		Assertions.assertEquals("SOME_ID", axonsType.getId());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.SOME_ID", axonsType.getQualifiedId());		
		Assertions.assertEquals(AxonsBaseType.CUSTOM.getBaseType(), axonsType.getParentType());	
		Assertions.assertEquals(AxonsBaseType.CUSTOM.asAxonsType().getId(), axonsType.getParentType().getId());	
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.SOME_ID", axonsType.toString());
	}
	
	@Test
	public void testCreateSubType() {
		AxonsType axonsSubType = AxonsType.createSubType(AxonsBaseType.CONVOLUTIONAL, "SOME_ID");
		Assertions.assertNotNull(axonsSubType);
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsSubType.getBaseType());
		Assertions.assertEquals("SOME_ID", axonsSubType.getId());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID", axonsSubType.getQualifiedId());
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsSubType.getParentType());	
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID", axonsSubType.toString());

	}
	
	@Test
	public void testCreateSubTypeWhenParentIsAxonsType() {
		AxonsType axonsSubType = AxonsType.createSubType(AxonsBaseType.CONVOLUTIONAL.asAxonsType(), "SOME_ID");
		Assertions.assertNotNull(axonsSubType);
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsSubType.getBaseType());
		Assertions.assertEquals("SOME_ID", axonsSubType.getId());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID", axonsSubType.getQualifiedId());
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL.asAxonsType(), axonsSubType.getParentType());	
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID", axonsSubType.toString());
	}
	
	@Test
	public void testCreateSubTypeWhenParentIsSubtypeAxonsType() {
		AxonsType parentType = AxonsType.createSubType(AxonsBaseType.CONVOLUTIONAL.asAxonsType(), "SOME_ID_1");
		AxonsType axonsSubType = AxonsType.createSubType(parentType, "SOME_ID_2");
		Assertions.assertNotNull(axonsSubType);
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsSubType.getBaseType());
		Assertions.assertEquals("SOME_ID_2", axonsSubType.getId());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID_1.SOME_ID_2", axonsSubType.getQualifiedId());
		Assertions.assertEquals(parentType, axonsSubType.getParentType());		
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID_1.SOME_ID_2", axonsSubType.toString());

	}
	
	@Test
	public void testCreateCustomBaseTypeNameClash() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {


			AxonsType.createCustomBaseType("CONVOLUTIONAL");

		});
	}
	
	@Test
	public void testConstructorWhenNotBaseType() {
		AxonsType axonsType = new AxonsType(AxonsBaseType.CONVOLUTIONAL, "SOME_ID", false, false);
		Assertions.assertEquals("SOME_ID", axonsType.getId());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID", axonsType.getQualifiedId());
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsType.getBaseType());
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsType.getParentType());
		Assertions.assertFalse(axonsType.isStandardBaseType());
		Assertions.assertFalse(axonsType.isCustomBaseType());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID", axonsType.toString());

	}
	
	@Test
	public void testConstructorWhenStandardBaseTypeAndIdDoesNotMatch() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {

			new AxonsType(AxonsBaseType.CONVOLUTIONAL, "SOME_ID", true, false);

		});
	}
	
	@Test
	public void testConstructorWhenStandardBaseTypeAndIdMatches() {
		AxonsType axonsType = new AxonsType(AxonsBaseType.CONVOLUTIONAL, "CONVOLUTIONAL", true, false);
		Assertions.assertEquals("CONVOLUTIONAL", axonsType.getId());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL", axonsType.getQualifiedId());
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsType.getBaseType());
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsType.getParentType());
		Assertions.assertTrue(axonsType.isStandardBaseType());
		Assertions.assertFalse(axonsType.isCustomBaseType());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL", axonsType.toString());

	}
	
	@Test
	public void testConstructorWhenStandardBaseTypeAndIdMatchesAndParentTypeIsAnAxonsTypeInstance() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {

			new AxonsType(AxonsBaseType.CONVOLUTIONAL.asAxonsType(), "CONVOLUTIONAL", true, false);

		});
	}
	
	@Test
	public void testConstructorWhenCustomBaseType() {
		AxonsType axonsType = new AxonsType(AxonsBaseType.CONVOLUTIONAL, "SOME_ID", false, true);
		Assertions.assertEquals("SOME_ID", axonsType.getId());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID", axonsType.getQualifiedId());
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsType.getBaseType());
		Assertions.assertEquals(AxonsBaseType.CONVOLUTIONAL, axonsType.getParentType());
		Assertions.assertFalse(axonsType.isStandardBaseType());
		Assertions.assertTrue(axonsType.isCustomBaseType());
		Assertions.assertEquals("org.ml4j.nn.axons.AxonsBaseType.CONVOLUTIONAL.SOME_ID", axonsType.toString());
	}
	
	@Test
	public void testHashCode() {
		AxonsType axonsType1 = new AxonsType(AxonsBaseType.CONVOLUTIONAL, "CONVOLUTIONAL", true, false);
		AxonsType axonsType2 = new AxonsType(AxonsBaseType.CONVOLUTIONAL, "CONVOLUTIONAL", true, false);
		Assertions.assertEquals(axonsType1.hashCode(), axonsType2.hashCode());
	}
	

	@Test
	public void testConstructorWhenCustomBaseTypeAndStandardBaseTypeAndNameMatches() {

		Assertions.assertThrows(IllegalArgumentException.class, () -> {

			new AxonsType(AxonsBaseType.CONVOLUTIONAL, "CONVOLUTIONAL", true, true);

		});
	}

}
