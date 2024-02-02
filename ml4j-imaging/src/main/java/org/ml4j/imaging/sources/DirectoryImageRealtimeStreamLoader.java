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
import java.io.File;
import java.io.IOException;
import java.util.Date;

import javax.imageio.ImageIO;

import org.ml4j.imaging.AbstractFrameSequenceSource;
import org.ml4j.imaging.FrameSequenceSource;
import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.labeling.IdFromLabelExtractor;
import org.ml4j.imaging.labeling.TimestampFromTimestampedFilenameGenerator;
import org.ml4j.imaging.targets.ImageDisplay;

/**
 * <p>
 * A FrameSequenceSource which loads sequences of BufferedImages from a
 * directory, loading the images in real-time using timestamps contained within
 * the file names. Ids for the buffered images are the timestamps associated
 * with each image
 * </p>
 *
 * @author Michael Lavelle
 */
public class DirectoryImageRealtimeStreamLoader extends AbstractFrameSequenceSource<SerializableBufferedImageAdapter, Long> implements
		FrameSequenceSource<SerializableBufferedImageAdapter, Long> {

	private File directory;

	private IdFromLabelExtractor<Long, String> idFromFileNameExtractor;

	/**
	 * <p>
	 * extractFrames.
	 * </p>
	 *
	 * @throws java.lang.InterruptedException
	 *             if any.
	 * @throws java.io.IOException
	 *             if any.
	 */
	public void extractFrames() throws InterruptedException, IOException {
		Long lastFrameTime = null;
		int delayMillis = 0;
		for (File file : directory.listFiles()) {
			BufferedImage bufferedImage = ImageIO.read(file);
			long frameTime = idFromFileNameExtractor.getIdFromLabel(file.getName());

			imageUpdate(new SerializableBufferedImageAdapter(bufferedImage), delayMillis);

			if (lastFrameTime != null) {
				delayMillis = (int) (frameTime - lastFrameTime);
			}

			lastFrameTime = frameTime;

		}
	}

	/**
	 * <p>
	 * main.
	 * </p>
	 *
	 * @param args
	 *            an array of {@link java.lang.String} objects.
	 * @throws java.lang.InterruptedException
	 *             if any.
	 * @throws java.io.IOException
	 *             if any.
	 */
	public static void main(String[] args) throws InterruptedException, IOException {
		DirectoryImageRealtimeStreamLoader loader = new DirectoryImageRealtimeStreamLoader(new Dimension(640, 360),
				new File("/Users/michael/webcamextract"));

		// Displaying the (decorated) images when they are extracted
		loader.addFrameUpdateListener(new ImageDisplay<Long>(loader.getFrameWidth(), loader.getFrameHeight()));

		loader.extractFrames();

	}

	/**
	 * <p>
	 * Constructor for DirectoryImageRealtimeStreamLoader.
	 * </p>
	 *
	 * @param size
	 *            a {@link java.awt.Dimension} object.
	 * @param directory
	 *            a {@link java.io.File} object.
	 */
	public DirectoryImageRealtimeStreamLoader(Dimension size, File directory) {
		super((int) size.getWidth(), (int) size.getHeight());
		this.directory = directory;
		this.idFromFileNameExtractor = new TimestampFromTimestampedFilenameGenerator();

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
	private void imageUpdate(SerializableBufferedImageAdapter image, int delayMillis) throws InterruptedException {
		long lastNotifyTime = new Date().getTime();
		frameUpdated(image, lastNotifyTime);

		if (new Date().getTime() < lastNotifyTime + delayMillis) {
			Thread.sleep(lastNotifyTime + delayMillis - new Date().getTime());
		}

	}

}
