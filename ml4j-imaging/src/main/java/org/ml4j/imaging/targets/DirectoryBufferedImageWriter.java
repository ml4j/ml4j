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
package org.ml4j.imaging.targets;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.ml4j.imaging.FrameUpdateListener;
import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.labeling.LabelFromIdExtractor;

/**
 * <p>
 * DirectoryBufferedImageWriter class.
 * </p>
 *
 * @author Michael Lavelle
 */
public class DirectoryBufferedImageWriter<ID> implements FrameUpdateListener<SerializableBufferedImageAdapter, ID> {

	private File dir;
	private LabelFromIdExtractor<ID, String> fileNameGenerator;

	/**
	 * <p>
	 * Constructor for DirectoryBufferedImageWriter.
	 * </p>
	 *
	 * @param dir
	 *            a {@link java.lang.String} object.
	 * @param fileNameGenerator
	 *            a {@link org.ml4j.imaging.labeling.LabelFromIdExtractor}
	 *            object.
	 */
	public DirectoryBufferedImageWriter(String dir, LabelFromIdExtractor<ID, String> fileNameGenerator) {
		this.dir = new File(dir);
		this.fileNameGenerator = fileNameGenerator;
	}

	/**
	 * <p>
	 * writeBufferedImage.
	 * </p>
	 *
	 * @param bufferedImage
	 *            a {@link java.awt.image.BufferedImage} object.
	 * @param fileName
	 *            a {@link java.lang.String} object.
	 * @throws java.io.IOException
	 *             if any.
	 */
	private void writeBufferedImage(BufferedImage bufferedImage, String fileName) throws IOException {
		// retrieve image
		File outputfile = new File(dir, fileName);
		ImageIO.write(bufferedImage, "jpg", outputfile);
	}

	/** {@inheritDoc} */
	@Override
	public void onFrameUpdate(SerializableBufferedImageAdapter frame, ID frameId) {
		try {
			writeBufferedImage(frame.getImage(), fileNameGenerator.getLabelFromId(frameId));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
