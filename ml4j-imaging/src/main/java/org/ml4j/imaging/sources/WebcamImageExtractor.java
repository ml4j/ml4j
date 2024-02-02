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
package org.ml4j.imaging.sources;

import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.util.Date;

import org.ml4j.imaging.AbstractFrameSequenceSource;
import org.ml4j.imaging.FrameSequenceSource;
import org.ml4j.imaging.SerializableBufferedImageAdapter;

import com.github.sarxos.webcam.Webcam;

/**
 * <p>
 * A BufferedImage FrameSequenceSource which Extracts images at a fixed rate
 * from a WebCam, setting the ids for the frames to be the captured image
 * timestamp
 * </p>
 *
 * @author Michael Lavelle
 */
public class WebcamImageExtractor extends AbstractFrameSequenceSource<SerializableBufferedImageAdapter, Long> implements
		FrameSequenceSource<SerializableBufferedImageAdapter, Long> {

	private Webcam webcam;

	/**
	 * <p>
	 * extractFrames.
	 * </p>
	 *
	 * @param frameCount
	 *            a int.
	 * @param delayMillis
	 *            a int.
	 * @throws java.lang.InterruptedException
	 *             if any.
	 */
	public void extractFrames(int frameCount, int delayMillis) throws InterruptedException {
		for (int i = 0; i < frameCount; i++) {
			BufferedImage webcamImage = webcam.getImage();
			imageUpdate(webcamImage, delayMillis);
		}
	}

	/**
	 * <p>
	 * Constructor for WebcamImageExtractor.
	 * </p>
	 *
	 * @param size
	 *            a {@link java.awt.Dimension} object.
	 */
	public WebcamImageExtractor(Dimension size) {
		super((int) size.getWidth(), (int) size.getHeight());
		webcam = Webcam.getDefault();
		webcam.setViewSize(size);
		webcam.open(true);

	}

	/**
	 * <p>
	 * imageUpdate.
	 * </p>
	 *
	 * @param image
	 *            a {@link java.awt.image.BufferedImage} object.
	 * @param delayMillis
	 *            a int.
	 * @throws java.lang.InterruptedException
	 *             if any.
	 */
	private void imageUpdate(BufferedImage image, int delayMillis) throws InterruptedException {
		long lastNotifyTime = new Date().getTime();
		frameUpdated(new SerializableBufferedImageAdapter(image), lastNotifyTime);
		if (new Date().getTime() < lastNotifyTime + delayMillis) {
			Thread.sleep(lastNotifyTime + delayMillis - new Date().getTime());
		}

	}

}
