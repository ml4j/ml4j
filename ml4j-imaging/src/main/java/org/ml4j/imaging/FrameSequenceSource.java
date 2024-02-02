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

/**
 * <p>
 * Defines a source of a sequence of image frames represented by instances of
 * type F and id of type ID, all with the same width and height.
 * </p>
 *
 * @author Michael Lavelle
 */
public interface FrameSequenceSource<F, ID> {

	/**
	 * <p>
	 * addFrameUpdateListener.
	 * </p>
	 *
	 * @param frameUpdateListener
	 *            a {@link org.ml4j.imaging.FrameUpdateListener} object.
	 */
	public void addFrameUpdateListener(FrameUpdateListener<F, ID> frameUpdateListener);

	/**
	 * <p>
	 * addFrameDecorator.
	 * </p>
	 *
	 * @param frameDecorator
	 *            a {@link org.ml4j.imaging.FrameDecorator} object.
	 */
	public void addFrameDecorator(FrameDecorator<F, ID> frameDecorator);

	/**
	 * <p>
	 * getFrameWidth.
	 * </p>
	 */
	public int getFrameWidth();

	/**
	 * <p>
	 * getFrameHeight.
	 * </p>
	 */
	public int getFrameHeight();
}
