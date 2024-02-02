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

import org.ml4j.imaging.tracking.Position3D;
import org.ml4j.mapping.LabeledData;

/**
 * <p>
 * Decorates a Point-labeled BufferedImage with a rectangle containing the
 * point, having specified width and height
 * </p>
 *
 * @author Michael Lavelle
 */
public class Position3DOverlayDecorator<ID> implements FrameDecorator<LabeledData<SerializableBufferedImageAdapter, Position3D>, ID> {

	private int pointWidth;
	private int pointHeight;

	/**
	 * <p>
	 * Constructor for AddPointDecorator.
	 * </p>
	 *
	 * @param pointWidth
	 *            a int.
	 * @param pointHeight
	 *            a int.
	 */
	public Position3DOverlayDecorator(int pointWidth, int pointHeight) {
		this.pointHeight = pointHeight;
		this.pointWidth = pointWidth;
	}

	/** {@inheritDoc} */
	@Override
	public LabeledData<SerializableBufferedImageAdapter, Position3D> decorateFrame(LabeledData<SerializableBufferedImageAdapter, Position3D> image, ID frameId) {

		if (image.getLabel() != null) {
			image.getData().getImage()
					.getGraphics()
					.drawRect((int) image.getLabel().getPointOnCamera().getX() - pointWidth / 2,
							(int) image.getLabel().getPointOnCamera().getY() - pointHeight / 2, pointWidth, pointHeight);
		}
		return image;
	}

	/** {@inheritDoc} */
	@Override
	public double getScaleFactor() {
		return 1d;
	}

}
