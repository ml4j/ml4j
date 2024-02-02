package org.ml4j.imaging;
/*
 * Copyright 2014 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.Serializable;

import org.ml4j.mapping.LabeledData;

/**
 * <p>LabeledFrameScalingDecorator class.</p>
 *
 * @author Michael Lavelle
 */
public class LabeledFrameScalingDecorator<ID,L extends Serializable> implements FrameDecorator<LabeledData<SerializableBufferedImageAdapter,L>, ID> {

	private double scaleFactor;
	
	/**
	 * <p>Constructor for LabeledFrameScalingDecorator.</p>
	 *
	 * @param scaleFactor a double.
	 */
	public LabeledFrameScalingDecorator(double scaleFactor)
	{
		this.scaleFactor = scaleFactor;
	}
	
	/** {@inheritDoc} */
	@Override
	public LabeledData<SerializableBufferedImageAdapter,L> decorateFrame(LabeledData<SerializableBufferedImageAdapter,L> frame, ID frameId) {
		int w = (int)( frame.getData().getImage().getWidth() * scaleFactor);
		int h = (int)(frame.getData().getImage().getHeight() * scaleFactor);
		
		return new LabeledData<SerializableBufferedImageAdapter,L> (new SerializableBufferedImageAdapter(getScaledImage(frame.getData().getImage(),w,h)),frame.getLabel());
	}
	
	 /**
	  * <p>getScaledImage.</p>
	  *
	  * @param srcImg a {@link java.awt.Image} object.
	  * @param w a int.
	  * @param h a int.
	  */
	 public static BufferedImage getScaledImage(Image srcImg, int w, int h){
	        BufferedImage resizedImg = new BufferedImage(w, h, BufferedImage.TRANSLUCENT);
	        Graphics2D g2 = resizedImg.createGraphics();
	        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
	        g2.drawImage(srcImg, 0, 0, w, h, null);
	        g2.dispose();
	        return resizedImg;
	    }

	/** {@inheritDoc} */
	@Override
	public double getScaleFactor() {		// TODO Auto-generated method stub
		return scaleFactor;
	}

}
