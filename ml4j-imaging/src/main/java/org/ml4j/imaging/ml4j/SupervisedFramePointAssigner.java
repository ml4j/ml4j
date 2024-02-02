package org.ml4j.imaging.ml4j;

import java.awt.Point;

import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.mapping.LabelAssigner;
import org.ml4j.mapping.LabeledData;

/**
 * <p>SupervisedFramePointAssigner class.</p>
 *
 * @author michael
 */
public class SupervisedFramePointAssigner implements LabelAssigner<SerializableBufferedImageAdapter,Point> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private ImageDisplay<Long> imageDisplay;
	
	/**
	 * <p>Constructor for SupervisedFramePointAssigner.</p>
	 *
	 * @param width a int.
	 * @param height a int.
	 */
	public SupervisedFramePointAssigner(int width,int height)
	{
		imageDisplay = new ImageDisplay<Long>(width,height);

	}
	
	/** {@inheritDoc} */
	@Override
	public LabeledData<SerializableBufferedImageAdapter, Point> assignLabel(SerializableBufferedImageAdapter data) {
		
		imageDisplay.onFrameUpdate(data, null);
		
		Point point = imageDisplay.nextMouseClick();
		return new LabeledData<SerializableBufferedImageAdapter,Point>(data,point);
	}

}
