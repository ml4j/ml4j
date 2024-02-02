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
package org.ml4j.imaging;

import java.util.ArrayList;
import java.util.List;

/**
 * <p>
 * Abstract FrameSequenceSource class.
 * </p>
 *
 * @author michael
 */
public abstract class AbstractFrameSequenceSource<F, ID> implements FrameSequenceSource<F, ID> {

	private List<FrameUpdateListener<F, ID>> frameUpdateListeners;
	private List<FrameDecorator<F, ID>> frameDecorators;

	private int width;
	private int height;

	/**
	 * <p>
	 * Constructor for AbstractFrameSequenceSource.
	 * </p>
	 *
	 * @param width
	 *            a int.
	 * @param height
	 *            a int.
	 */
	public AbstractFrameSequenceSource(int width, int height) {
		this.frameUpdateListeners = new ArrayList<FrameUpdateListener<F, ID>>();
		this.frameDecorators = new ArrayList<FrameDecorator<F, ID>>();
		this.width = width;
		this.height = height;
	}

	/** {@inheritDoc} */
	@Override
	public void addFrameUpdateListener(FrameUpdateListener<F, ID> frameUpdateListener) {
		frameUpdateListeners.add(frameUpdateListener);
	}

	/**
	 * <p>
	 * frameUpdated.
	 * </p>
	 *
	 * @param frame
	 *            a F object.
	 * @param frameId
	 *            a ID object.
	 * @param frameId
	 *            a ID object.
	 */
	protected void frameUpdated(F frame, ID frameId) {
		for (FrameDecorator<F, ID> decorator : frameDecorators) {
			frame = decorator.decorateFrame(frame, frameId);
			this.width = (int)(width * decorator.getScaleFactor());
			this.height = (int)(height * decorator.getScaleFactor());

		}
		for (FrameUpdateListener<F, ID> frameUpdateListener : frameUpdateListeners) {
			frameUpdateListener.onFrameUpdate(frame, frameId);
		}

	}

	/** {@inheritDoc} */
	@Override
	public void addFrameDecorator(FrameDecorator<F, ID> frameDecorator) {
		frameDecorators.add(frameDecorator);
	}

	/** {@inheritDoc} */
	@Override
	public int getFrameWidth() {
		return width;
	}

	/** {@inheritDoc} */
	@Override
	public int getFrameHeight() {
		return height;
	}

}
