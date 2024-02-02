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
package org.ml4j.imaging.examples;

import java.awt.Dimension;
import java.io.IOException;

import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.labeling.LabeledDataFrameUpdateListenerAdapter;
import org.ml4j.imaging.labeling.LabeledPosition3DLogger;
import org.ml4j.imaging.labeling.LabelingFrameSequenceSource;
import org.ml4j.imaging.labeling.TimestampedIdDelayLogger;
import org.ml4j.imaging.sources.Position3DLabelingFrameSequenceSourceFactory;
import org.ml4j.imaging.sources.WebcamImageExtractor;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.imaging.tracking.FIRVelocityFilter;
import org.ml4j.imaging.tracking.MovingTargetPositionEstimator;
import org.ml4j.imaging.tracking.PointAsPosition3DEstimator;
import org.ml4j.imaging.tracking.Position3D;
import org.ml4j.imaging.tracking.VelocityFilter;
import org.ml4j.imaging.tracking.labeling.MotionCOGPosition3DLabelAssigner;
import org.ml4j.mapping.LabelAssigner;
import org.ml4j.mapping.LabeledData;

/**
 * <p>
 * WebcamMovingTargetPositionEstimatorExample class.
 * </p>
 *
 * @author Michael Lavelle
 */
public class WebcamMovingTargetPositionEstimatorExample {

	private MovingTargetPositionEstimator movingTargetPositionEstimator;
	
	// Indicates whether to display the point-labeled images in a JDialog
	private boolean displayLabeledImages = false;
	
	/**
	 * <p>main.</p>
	 *
	 * @param args an array of {@link java.lang.String} objects.
	 * @throws java.lang.InterruptedException if any.
	 * @throws java.io.IOException if any.
	 */
	public static void main(String[] args) throws InterruptedException, IOException
	{
		WebcamMovingTargetPositionEstimatorExample example
		 = new WebcamMovingTargetPositionEstimatorExample();
		example.runPositionEstimation();
	}
	
	/**
	 * <p>Constructor for WebcamMovingTargetPositionEstimatorExample.</p>
	 */
	public WebcamMovingTargetPositionEstimatorExample()
	{
		VelocityFilter velocityFilter = new FIRVelocityFilter(0,5);
		movingTargetPositionEstimator = new MovingTargetPositionEstimator(velocityFilter,velocityFilter,velocityFilter);
	}
	
	/**
	 * <p>runPositionEstimation.</p>
	 *
	 * @throws java.lang.InterruptedException if any.
	 * @throws java.io.IOException if any.
	 */
	public void runPositionEstimation() throws InterruptedException, IOException
	{
		
		// Create a frame sequence source for our webcam
		WebcamImageExtractor webcamImageExtractor
		 = new WebcamImageExtractor(new Dimension(640,480));
		

		// Create a motion COG label assigner
		LabelAssigner<SerializableBufferedImageAdapter, Position3D> motionCOGLabelAssigner = // new
																		// NoOpPointLabelAssigner();

				
				
		new MotionCOGPosition3DLabelAssigner(webcamImageExtractor.getFrameWidth(), webcamImageExtractor.getFrameHeight(),
				true, new PointAsPosition3DEstimator());

		// Create a point-labeling frame sequence source for the webcam images and motion cog label assigner 
		// Decorate the images by adding a tracking box for the labeled point
		Position3DLabelingFrameSequenceSourceFactory<Long> application = new Position3DLabelingFrameSequenceSourceFactory<Long>(
				webcamImageExtractor, motionCOGLabelAssigner, 200, 300, true);

		LabelingFrameSequenceSource<SerializableBufferedImageAdapter, Long, Position3D> labeledFrameSequenceSource = application
				.getLabeledFrameSequenceSource();

		// Configure the moving target position estimator to listen to the motion-COG-labeled frame sequence source
		labeledFrameSequenceSource.addFrameUpdateListener(movingTargetPositionEstimator);

		labeledFrameSequenceSource
				.addFrameUpdateListener(new TimestampedIdDelayLogger<LabeledData<SerializableBufferedImageAdapter, Position3D>>());

		// Display the (decorated) images when they are extracted
		if (displayLabeledImages)
		{
			labeledFrameSequenceSource
					.addFrameUpdateListener(new LabeledDataFrameUpdateListenerAdapter<SerializableBufferedImageAdapter, Position3D, Long>(
							new ImageDisplay<Long>(labeledFrameSequenceSource.getFrameWidth(), labeledFrameSequenceSource
								.getFrameHeight())));
		}
		else
		{
			labeledFrameSequenceSource.addFrameUpdateListener(new LabeledPosition3DLogger<SerializableBufferedImageAdapter,Long>());

		}

		webcamImageExtractor.extractFrames(300,20);

	
	}

	
	


}
