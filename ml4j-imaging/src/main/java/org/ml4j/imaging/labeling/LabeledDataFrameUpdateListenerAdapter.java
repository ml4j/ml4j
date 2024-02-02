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

import org.ml4j.imaging.FrameUpdateListener;
import org.ml4j.mapping.LabeledData;

/**
 * <p>
 * LabeledDataFrameUpdateListenerAdapter class.
 * </p>
 *
 * @author Michael Lavelle
 */
public class LabeledDataFrameUpdateListenerAdapter<T extends Serializable, L extends Serializable, ID> implements FrameUpdateListener<LabeledData<T, L>, ID> {

	private FrameUpdateListener<T, ID> frameUpdateListener;

	/**
	 * <p>
	 * Constructor for LabeledDataFrameUpdateListenerAdapter.
	 * </p>
	 *
	 * @param frameUpdateListener
	 *            a {@link org.ml4j.imaging.FrameUpdateListener} object.
	 */
	public LabeledDataFrameUpdateListenerAdapter(FrameUpdateListener<T, ID> frameUpdateListener) {
		this.frameUpdateListener = frameUpdateListener;
	}

	/** {@inheritDoc} */
	@Override
	public void onFrameUpdate(LabeledData<T, L> frame, ID frameId) {

		frameUpdateListener.onFrameUpdate(frame.getData(), frameId);

	}

}
