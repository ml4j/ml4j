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
package org.ml4j.imaging.tracking;

import java.awt.Dimension;
import java.io.File;
import java.io.IOException;
import java.net.URL;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.ml4j.imaging.FrameScalingDecorator;
import org.ml4j.imaging.FrameSequenceSource;
import org.ml4j.imaging.LabeledFrameScalingDecorator;
import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.labeling.LabeledDataFrameUpdateListenerAdapter;
import org.ml4j.imaging.labeling.LabeledPosition3DLogger;
import org.ml4j.imaging.labeling.LabelingFrameSequenceSource;
import org.ml4j.imaging.labeling.TimestampedIdDelayLogger;
import org.ml4j.imaging.sources.DirectoryImageRealtimeStreamLoader;
import org.ml4j.imaging.sources.Position3DLabelingFrameSequenceSourceFactory;
import org.ml4j.imaging.sources.WebcamImageExtractor;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.imaging.tracking.labeling.MotionCOGPosition3DLabelAssigner;
import org.ml4j.mapping.LabelAssigner;
import org.ml4j.mapping.LabeledData;

/**
 * <p>
 * WebcamMovingTargetPositionEstimatorIntegrationTest class.
 * </p>
 *
 * @author Michael Lavelle
 */

@Disabled
public class WebcamMovingTargetPositionEstimatorIntegrationTest {

    private MovingTargetPositionEstimator movingTargetPositionEstimator;

    // Indicates whether to use real webcam images or mock ones loaded in realtime from directory
    private boolean useMockWebcamImages = true;

    // Indicates whether to display the point-labeled images in a JDialog
    private boolean displayLabeledImages = false;

    // Whether to scale the source images, at a possible cost to accuracy (TBC) but
    // this increases motion detection speed by a factor of 1/(scaleFactor)^2.
    private boolean scaleSourceImages = true;

    private double scaleFactor = 0.5d;

    public WebcamMovingTargetPositionEstimatorIntegrationTest()
    {
        VelocityFilter velocityFilter = new FIRVelocityFilter(0,5);
        movingTargetPositionEstimator = new MovingTargetPositionEstimator(velocityFilter,velocityFilter,velocityFilter);
    }


    private FrameSequenceSource<SerializableBufferedImageAdapter,Long> createMockWebcamFrameSequenceSource()
    {
        // Setup (mock) webcam frame sequence source, loading images from classpath directory
        URL url = WebcamMovingTargetPositionEstimatorIntegrationTest.class.getClassLoader().getResource("./webcamimages");
        File imagesDir = new File(url.getFile());

        DirectoryImageRealtimeStreamLoader mockWebcamImageExtractor = new DirectoryImageRealtimeStreamLoader(new Dimension(
                640, 360), imagesDir);

        return mockWebcamImageExtractor;
    }

    private FrameSequenceSource<SerializableBufferedImageAdapter, Long> createWebcamFrameSequenceSource() {
        return new WebcamImageExtractor(new Dimension(640,480));
    }

    @Test
    public void testPositionEstimation() throws InterruptedException, IOException
    {

        // Create a frame sequence source for our webcam
        FrameSequenceSource<SerializableBufferedImageAdapter,Long> webcamImageExtractor
                = useMockWebcamImages ? createMockWebcamFrameSequenceSource()
                : createWebcamFrameSequenceSource();

        if (scaleSourceImages)
        {

            // Scale down the source images - this speeds up the motion detection processing by a
            // factor of just under 4. but seems to retain much of the accuracy - TBC just how much
            // accururacy is lost?
            webcamImageExtractor.addFrameDecorator(new FrameScalingDecorator<Long>(scaleFactor));
        }

        //Commenting out no op label assigner used for benchmarking
        // LabelAssigner<BufferedImage, Point> motionCOGLabelAssigner = new NoOpPointLabelAssigner();


        // Create a motion COG label assigner instead

        // Decide whether the centre of gravity of motion is sticky - ie. remains labeled as
        // such until the next cog update.
        boolean stickyCog = true;
        LabelAssigner<SerializableBufferedImageAdapter, Position3D> motionCOGLabelAssigner =
                new MotionCOGPosition3DLabelAssigner(webcamImageExtractor.getFrameWidth(), webcamImageExtractor.getFrameHeight(),
                        stickyCog,new PointAsPosition3DEstimator());

        // Create a point-labeling frame sequence source for the webcam images and motion cog label assigner
        // Decorate the images by adding a tracking box for the labeled point
        Position3DLabelingFrameSequenceSourceFactory<Long> application = new Position3DLabelingFrameSequenceSourceFactory<Long>(
                webcamImageExtractor, motionCOGLabelAssigner, 10, 10, true);

        LabelingFrameSequenceSource<SerializableBufferedImageAdapter, Long, Position3D> labeledFrameSequenceSource = application
                .getLabeledFrameSequenceSource();

        // Configure the moving target position estimator to listen to the motion-COG-labeled frame sequence source
        labeledFrameSequenceSource.addFrameUpdateListener(movingTargetPositionEstimator);

        labeledFrameSequenceSource
                .addFrameUpdateListener(new TimestampedIdDelayLogger<LabeledData<SerializableBufferedImageAdapter, Position3D>>());

        // Display the (decorated) images when they are extracted
        if (displayLabeledImages)
        {
            // Optionally resize the image back to original proportions for display
            if (scaleSourceImages)
            {
                double resizeScale = 1d/scaleFactor;
                labeledFrameSequenceSource.addFrameDecorator(new LabeledFrameScalingDecorator<Long,Position3D>(resizeScale));
            }
            // Display
            labeledFrameSequenceSource
                    .addFrameUpdateListener(new LabeledDataFrameUpdateListenerAdapter<SerializableBufferedImageAdapter, Position3D, Long>(
                            new ImageDisplay<Long>(labeledFrameSequenceSource.getFrameWidth(), labeledFrameSequenceSource
                                    .getFrameHeight())));



        }
        else
        {
            labeledFrameSequenceSource.addFrameUpdateListener(new LabeledPosition3DLogger<SerializableBufferedImageAdapter,Long>());

        }

        if (useMockWebcamImages)
        {
            // Load images from directory, in real-time according to their labeled timestamps
            ((DirectoryImageRealtimeStreamLoader)webcamImageExtractor).extractFrames();
        }
        else
        {
            // Extract 300 frames from webcam at a min frame delay of 20 ms
            ((WebcamImageExtractor)webcamImageExtractor).extractFrames(300,20);

        }
    }





}