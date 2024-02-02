package org.ml4j.nn.neurons;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class NeuronsTest {

	@Test
	public void testProperties() {
		
		Neurons neurons1 = new Neurons(5, false);
		
		Assertions.assertFalse(neurons1.hasBiasUnit());
		Assertions.assertEquals(5, neurons1.getNeuronCountExcludingBias());
		Assertions.assertEquals(5, neurons1.getNeuronCountIncludingBias());
		
		Neurons neurons2 = new Neurons(5, false);
		Assertions.assertFalse(neurons2.hasBiasUnit());
		Assertions.assertEquals(5, neurons2.getNeuronCountExcludingBias());
		Assertions.assertEquals(5, neurons2.getNeuronCountIncludingBias());
		
		Neurons neurons3 = new Neurons(5, true);
		Assertions.assertTrue(neurons3.hasBiasUnit());
		Assertions.assertEquals(5, neurons3.getNeuronCountExcludingBias());
		Assertions.assertEquals(6, neurons3.getNeuronCountIncludingBias());
		
		Neurons neurons4 = new Neurons(6, false);
		Assertions.assertFalse(neurons4.hasBiasUnit());
		Assertions.assertEquals(6, neurons4.getNeuronCountExcludingBias());
		Assertions.assertEquals(6, neurons4.getNeuronCountIncludingBias());	
		
		Assertions.assertEquals(neurons1, neurons1);
		Assertions.assertEquals(neurons1.hashCode(), neurons1.hashCode());
		
		Assertions.assertNotNull(neurons1.toString());
	}
}
