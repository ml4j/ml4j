package org.ml4j.nn.axons;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.ml4j.nn.neurons.Neurons3D;

public class Axons3DConfigTest {
	
	@Test
	public void testBuilderWithFilterHeightAndWidthSet() {
		Axons3DConfig config = new Axons3DConfig(new Neurons3D(3, 3, 1, false), new Neurons3D(3,3,1,false));
		config.withFilterHeight(3).withFilterWidth(3)
			.withPaddingHeight(1).withPaddingWidth(1).withStrideHeight(1).withStrideWidth(1);
		
		Assertions.assertEquals((int)3, (int)config.getFilterHeight());
		Assertions.assertEquals((int)3, (int)config.getFilterWidth());
		Assertions.assertEquals((int)1, (int)config.getPaddingHeight());
		Assertions.assertEquals((int)1, (int)config.getPaddingWidth());
		Assertions.assertEquals((int)1, (int)config.getStrideHeight());
		Assertions.assertEquals((int)1, (int)config.getStrideWidth());

		Axons3DConfig dupConfig = config.dup();
		
		Assertions.assertEquals((int)3, (int)dupConfig.getFilterHeight());
		Assertions.assertEquals((int)3, (int)dupConfig.getFilterWidth());
		Assertions.assertEquals((int)1, (int)dupConfig.getPaddingHeight());
		Assertions.assertEquals((int)1, (int)dupConfig.getPaddingWidth());
		Assertions.assertEquals((int)1, (int)dupConfig.getStrideHeight());
		Assertions.assertEquals((int)1, (int)dupConfig.getStrideWidth());

		Assertions.assertTrue(config.equals(dupConfig));

	}
	
	@Test
	public void testBuilderWithoutFilterHeightAndWidthSet() {
		Axons3DConfig config = new Axons3DConfig(new Neurons3D(3, 3, 1, false), new Neurons3D(3,3,1,false))
				.withPaddingHeight(1).withPaddingWidth(1).withStrideHeight(1).withStrideWidth(1);
		
		Assertions.assertEquals((int)3, (int)config.getFilterHeight());
		Assertions.assertEquals((int)3, (int)config.getFilterWidth());
		Assertions.assertEquals((int)1, (int)config.getPaddingHeight());
		Assertions.assertEquals((int)1, (int)config.getPaddingWidth());
		Assertions.assertEquals((int)1, (int)config.getStrideHeight());
		Assertions.assertEquals((int)1, (int)config.getStrideWidth());

		Axons3DConfig dupConfig = config.dup();
		
		Assertions.assertEquals((int)3, (int)dupConfig.getFilterHeight());
		Assertions.assertEquals((int)3, (int)dupConfig.getFilterWidth());
		Assertions.assertEquals((int)1, (int)dupConfig.getPaddingHeight());
		Assertions.assertEquals((int)1, (int)dupConfig.getPaddingWidth());
		Assertions.assertEquals((int)1, (int)dupConfig.getStrideHeight());
		Assertions.assertEquals((int)1, (int)dupConfig.getStrideWidth());

		Assertions.assertTrue(config.equals(dupConfig));

	}

}
