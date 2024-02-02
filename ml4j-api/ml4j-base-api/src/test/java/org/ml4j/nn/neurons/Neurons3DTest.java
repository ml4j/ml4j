package org.ml4j.nn.neurons;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class Neurons3DTest {

	@Test
	public void testProperties() {
		
		Neurons3D neurons1 = new Neurons3D(2, 3, 5, false);
		
		Assertions.assertFalse(neurons1.hasBiasUnit());
		Assertions.assertEquals(30, neurons1.getNeuronCountExcludingBias());
		Assertions.assertEquals(30, neurons1.getNeuronCountIncludingBias());
		Assertions.assertEquals(2, neurons1.getWidth());
		Assertions.assertEquals(3, neurons1.getHeight());
		Assertions.assertEquals(5, neurons1.getDepth());
		

		Neurons3D neurons2 = new Neurons3D(2, 3, 5, false);
		Assertions.assertFalse(neurons2.hasBiasUnit());
		Assertions.assertEquals(30, neurons2.getNeuronCountExcludingBias());
		Assertions.assertEquals(30, neurons2.getNeuronCountIncludingBias());
		Assertions.assertEquals(2, neurons2.getWidth());
		Assertions.assertEquals(3, neurons2.getHeight());
		Assertions.assertEquals(5, neurons2.getDepth());
		
		Neurons3D neurons3 = new Neurons3D(2, 3, 5, true);
		Assertions.assertTrue(neurons3.hasBiasUnit());
		Assertions.assertEquals(30, neurons3.getNeuronCountExcludingBias());
		Assertions.assertEquals(31, neurons3.getNeuronCountIncludingBias());
		Assertions.assertEquals(2, neurons3.getWidth());
		Assertions.assertEquals(3, neurons3.getHeight());
		Assertions.assertEquals(5, neurons3.getDepth());	
		
		Assertions.assertEquals(neurons1, neurons1);
		Assertions.assertEquals(neurons1.hashCode(), neurons1.hashCode());
		
		Assertions.assertNotNull(neurons1.toString());

	}
}
