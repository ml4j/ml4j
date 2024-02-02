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
 * Defines components which decorate frames of type F and id of type ID
 * </p>
 *
 * @author michael
 */
public interface FrameDecorator<F, ID> {

	/**
	 * <p>
	 * decorateFrame.
	 * </p>
	 *
	 * @param frame
	 *            a F object.
	 * @param frameId
	 *            a ID object.
	 * @param frameId
	 *            a ID object.
	 */
	public F decorateFrame(F frame, ID frameId);

	/**
	 * <p>getScaleFactor.</p>
	 */
	public double getScaleFactor();

}
