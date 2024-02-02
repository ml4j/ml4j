package org.ml4j.imaging.examples;

import java.awt.Dimension;

import org.ml4j.imaging.labeling.FilenameFromTimestampGenerator;
import org.ml4j.imaging.sources.WebcamImageExtractor;
import org.ml4j.imaging.targets.DirectoryBufferedImageWriter;

/**
 * <p>WebcamImageWriterExample class.</p>
 *
 * @author michael
 */
public class WebcamImageWriterExample {

	/**
	 * <p>main.</p>
	 *
	 * @param args an array of {@link java.lang.String} objects.
	 * @throws java.lang.Throwable if any.
	 */
	public static void main(String[] args) throws Throwable
	{
		
		if (args.length !=1 )
		{
			throw new IllegalStateException("Please supply write directory path as a program argument");
		}
		
		String directoryPath = args[0];
		
		WebcamImageExtractor webcamImageExtractor = new WebcamImageExtractor(new Dimension(640,480));
		
		// Displaying the images when they are extracted
		webcamImageExtractor.addFrameUpdateListener(
			new DirectoryBufferedImageWriter<Long>(directoryPath,new FilenameFromTimestampGenerator("jpg")));
	
		// Extract 300 frames with a min delay of 100ms
		webcamImageExtractor.extractFrames(300, 100);
		
				
	}
	
}
