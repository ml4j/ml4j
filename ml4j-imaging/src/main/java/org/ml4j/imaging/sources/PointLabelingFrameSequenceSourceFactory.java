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

import java.awt.Point;

import org.ml4j.imaging.FrameSequenceSource;
import org.ml4j.imaging.PointOverlayDecorator;
import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.labeling.LabelFromIdExtractor;
import org.ml4j.imaging.labeling.LabelingFrameSequenceSource;
import org.ml4j.imaging.labeling.LabelingFrameUpdateListener;
import org.ml4j.mapping.LabelAssigner;

/**
 * <p>
 * Creates a source for a sequence of Point-labeled BufferedImages, given an
 * underlying BufferedImageSource and a Point-label assigner
 * </p>
 *
 * @author Michael Lavelle
 */
public class PointLabelingFrameSequenceSourceFactory<ID> {

	private LabelingFrameSequenceSource<SerializableBufferedImageAdapter, ID, Point> labeledFrameSequenceSource;

	private LabelAssigner<SerializableBufferedImageAdapter, Point> labelAssigner;
	private LabelFromIdExtractor<ID, Point> labelFromIdExtractor;
	private boolean decorateImages;
	private int pointWidth = 20;
	private int pointHeight = 20;

	/**
	 * <p>
	 * Constructor for PointLabelingSourceWrapper.
	 * </p>
	 *
	 * @param frameSequenceSource
	 *            a {@link org.ml4j.imaging.FrameSequenceSource} object.
	 * @param labelAssigner
	 *            a {@link org.ml4j.LabelAssigner} object.
	 * @param pointWidth
	 *            a int.
	 * @param pointHeight
	 *            a int.
	 * @param decorateImages
	 *            a boolean.
	 */
	public PointLabelingFrameSequenceSourceFactory(FrameSequenceSource<SerializableBufferedImageAdapter, ID> frameSequenceSource,
			LabelAssigner<SerializableBufferedImageAdapter, Point> labelAssigner, int pointWidth, int pointHeight, boolean decorateImages) {
		this.labelAssigner = labelAssigner;
		this.decorateImages = decorateImages;
		this.pointHeight = pointHeight;
		this.pointWidth = pointWidth;
		setupPointLabeling(frameSequenceSource);

	}

	/**
	 * <p>
	 * Constructor for PointLabelingSourceWrapper.
	 * </p>
	 *
	 * @param frameSequenceSource
	 *            a {@link org.ml4j.imaging.FrameSequenceSource} object.
	 * @param labelFromIdExtractor
	 *            a {@link org.ml4j.imaging.labeling.LabelFromIdExtractor}
	 *            object.
	 * @param pointWidth
	 *            a int.
	 * @param pointHeight
	 *            a int.
	 * @param decorateImages
	 *            a boolean.
	 */
	public PointLabelingFrameSequenceSourceFactory(FrameSequenceSource<SerializableBufferedImageAdapter, ID> frameSequenceSource,
			LabelFromIdExtractor<ID, Point> labelFromIdExtractor, int pointWidth, int pointHeight,
			boolean decorateImages) {
		this.labelFromIdExtractor = labelFromIdExtractor;
		this.decorateImages = decorateImages;
		this.pointHeight = pointHeight;
		this.pointWidth = pointWidth;
		setupPointLabeling(frameSequenceSource);

	}

	/**
	 * <p>
	 * Constructor for PointLabelingSourceWrapper.
	 * </p>
	 *
	 * @param frameSequenceSource
	 *            a {@link org.ml4j.imaging.FrameSequenceSource} object.
	 */
	public PointLabelingFrameSequenceSourceFactory(FrameSequenceSource<SerializableBufferedImageAdapter, ID> frameSequenceSource) {
		setupPointLabeling(frameSequenceSource);
	}

	/**
	 * <p>
	 * Getter for the field <code>labeledFrameSequenceSource</code>.
	 * </p>
	 */
	public LabelingFrameSequenceSource<SerializableBufferedImageAdapter, ID, Point> getLabeledFrameSequenceSource() {
		return labeledFrameSequenceSource;
	}

	/**
	 * <p>
	 * createPointLabeledFrameSequenceSource.
	 * </p>
	 *
	 * @param w
	 *            a int.
	 * @param h
	 *            a int.
	 */
	private LabelingFrameSequenceSource<SerializableBufferedImageAdapter, ID, Point> createPointLabeledFrameSequenceSource(int w, int h) {
 
		LabelingFrameSequenceSource<SerializableBufferedImageAdapter, ID, Point> labeledFrameSequenceSource = labelAssigner != null ? new LabelingFrameUpdateListener<SerializableBufferedImageAdapter, ID, Point>(
				w, h, labelAssigner) : new LabelingFrameUpdateListener<SerializableBufferedImageAdapter, ID, Point>(w, h,
				labelFromIdExtractor);
		if (decorateImages) {
			labeledFrameSequenceSource.addFrameDecorator(new PointOverlayDecorator<ID>(pointWidth, pointHeight));
		}
		return labeledFrameSequenceSource;
	}

	/**
	 * <p>
	 * setupPointLabeling.
	 * </p>
	 *
	 * @param frameSequenceSource
	 *            a {@link org.ml4j.imaging.FrameSequenceSource} object.
	 */
	public void setupPointLabeling(FrameSequenceSource<SerializableBufferedImageAdapter, ID> frameSequenceSource) {

		// Create a labeling frame sequence source which labels and decorates
		// the images
		this.labeledFrameSequenceSource = createPointLabeledFrameSequenceSource(frameSequenceSource.getFrameWidth(),
				frameSequenceSource.getFrameHeight());

		// Attach the label frame sequence source to the webcam image extractor
		frameSequenceSource.addFrameUpdateListener(labeledFrameSequenceSource);

	}

}
