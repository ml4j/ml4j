package org.ml4j.imaging.examples;

import java.awt.Dimension;
import java.awt.Point;

import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.labeling.LabeledDataFrameUpdateListenerAdapter;
import org.ml4j.imaging.labeling.LabeledPointLogger;
import org.ml4j.imaging.labeling.LabelingFrameSequenceSource;
import org.ml4j.imaging.ml4j.SupervisedFramePointAssigner;
import org.ml4j.imaging.sources.PointLabelingFrameSequenceSourceFactory;
import org.ml4j.imaging.sources.WebcamImageExtractor;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.mapping.LabelAssigner;

/**
 * <p>SupervisedWebcamImagePointLabelingSourceExample class.</p>
 *
 * @author michael
 */
public class SupervisedWebcamImagePointLabelingSourceExample {
	
	/**
	 * <p>main.</p>
	 *
	 * @param args an array of {@link java.lang.String} objects.
	 * @throws java.lang.Throwable if any.
	 */
	public static void main(String[] args) throws Throwable
	{
		WebcamImageExtractor webcamImageExtractor = new WebcamImageExtractor(new Dimension(640,480));
				
		LabelAssigner<SerializableBufferedImageAdapter,Point> supervisor
		 = new SupervisedFramePointAssigner(webcamImageExtractor.getFrameWidth(),webcamImageExtractor.getFrameHeight());
	
		
		PointLabelingFrameSequenceSourceFactory<Long> labeledFrameSequenceSourceFactory = new PointLabelingFrameSequenceSourceFactory<Long>(webcamImageExtractor,supervisor,20,20,true);
		
		LabelingFrameSequenceSource<SerializableBufferedImageAdapter, Long, Point> labeledFrameSequenceSource = labeledFrameSequenceSourceFactory.getLabeledFrameSequenceSource();
		
		// Listening to the labeling frame sequence source, printing out the labels when they are extracted
		labeledFrameSequenceSource.addFrameUpdateListener(new LabeledPointLogger<SerializableBufferedImageAdapter,Long>());
		
		// Displaying the (decorated) images when they are extracted
		labeledFrameSequenceSource.addFrameUpdateListener(
				new LabeledDataFrameUpdateListenerAdapter<SerializableBufferedImageAdapter,Point,Long>(new ImageDisplay<Long>(labeledFrameSequenceSource.getFrameWidth(),labeledFrameSequenceSource.getFrameHeight())));
	
		// Saving
		//String directoryPathToWriteTo = null;
		//labeledFrameSequenceSource.addFrameUpdateListener(
			//	new DirectoryLabeledBufferedImageWriter<Long,Point>(directoryPathToWriteTo,new FilenameFromTimestampAndPointGenerator("jpg")));
	
		
		// Extract 300 frames with a min delay of 20ms
		webcamImageExtractor.extractFrames(300, 20);
		
				
	}

}
