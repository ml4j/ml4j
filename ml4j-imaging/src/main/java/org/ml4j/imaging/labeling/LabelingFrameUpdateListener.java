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
package org.ml4j.imaging.labeling;

import java.io.Serializable;

import org.ml4j.imaging.AbstractFrameSequenceSource;
import org.ml4j.imaging.FrameUpdateListener;
import org.ml4j.mapping.LabelAssigner;
import org.ml4j.mapping.LabeledData;

/**
 * <p>
 * A LabelingFrameSequenceSource, can be registered with a FrameSequence source of frames of type F, and on detection of a frame update
 * assigns a label of type L to those frames.
 * </p>
 *
 * @author Michael Lavelle
 */
public class LabelingFrameUpdateListener<F extends Serializable, ID, L extends Serializable> extends AbstractFrameSequenceSource<LabeledData<F, L>, ID> implements
		FrameUpdateListener<F, ID>, LabelingFrameSequenceSource<F, ID, L> {

	private LabelAssigner<F, L> labelAssigner;
	private LabelFromIdExtractor<ID, L> labelFromIdExtractor;

	/**
	 * <p>
	 * Constructor for LabelingFrameUpdateListener.
	 * </p>
	 *
	 * @param width
	 *            a int.
	 * @param height
	 *            a int.
	 * @param labelAssigner
	 *            a {@link org.ml4j.LabelAssigner} object.
	 */
	public LabelingFrameUpdateListener(int width, int height, LabelAssigner<F, L> labelAssigner) {
		super(width, height);
		this.labelAssigner = labelAssigner;
	}

	/**
	 * <p>
	 * Constructor for LabelingFrameUpdateListener.
	 * </p>
	 *
	 * @param width
	 *            a int.
	 * @param height
	 *            a int.
	 * @param labelFromIdExtractor
	 *            a {@link org.ml4j.imaging.labeling.LabelFromIdExtractor}
	 *            object.
	 */
	public LabelingFrameUpdateListener(int width, int height, LabelFromIdExtractor<ID, L> labelFromIdExtractor) {
		super(width, height);
		this.labelFromIdExtractor = labelFromIdExtractor;
	}

	/** {@inheritDoc} */
	@Override
	public void onFrameUpdate(F frame, ID frameId) {

		LabeledData<F, L> labeledData = labelAssigner != null ? labelAssigner.assignLabel(frame)
				: new LabeledData<F, L>(frame, labelFromIdExtractor.getLabelFromId(frameId));
		frameUpdated(labeledData, frameId);
	}

}
