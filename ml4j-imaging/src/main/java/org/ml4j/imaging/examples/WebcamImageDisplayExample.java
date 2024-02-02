package org.ml4j.imaging.examples;

import java.awt.Dimension;

import org.ml4j.imaging.sources.WebcamImageExtractor;
import org.ml4j.imaging.targets.ImageDisplay;

/**
 * <p>WebcamImageDisplayExample class.</p>
 *
 * @author michael
 */
public class WebcamImageDisplayExample {

	/**
	 * <p>main.</p>
	 *
	 * @param args an array of {@link java.lang.String} objects.
	 * @throws java.lang.Throwable if any.
	 */
	public static void main(String[] args) throws Throwable
	{
		WebcamImageExtractor webcamImageExtractor = new WebcamImageExtractor(new Dimension(640,480));
		
		
		// Displaying the images when they are extracted
		webcamImageExtractor.addFrameUpdateListener(
			new ImageDisplay<Long>(webcamImageExtractor.getFrameWidth(),webcamImageExtractor.getFrameHeight()));
	
		
		// Extract 300 frames with a min delay of 20ms
		webcamImageExtractor.extractFrames(300, 20);
		
				
	}
	
}
