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
package org.ml4j.imaging.tracking.labeling;

import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.tracking.PointToPosition3DEstimator;
import org.ml4j.imaging.tracking.Position3D;
import org.ml4j.imaging.tracking.sarxos.BufferedImageStreamMotionDetector;
import org.ml4j.imaging.tracking.sarxos.BufferedImageStreamMotionEvent;
import org.ml4j.imaging.tracking.sarxos.BufferedImageStreamMotionListener;
import org.ml4j.mapping.LabelAssigner;
import org.ml4j.mapping.LabeledData;

/**
 * <p>
 * MotionCOGPointLabelAssigner class.
 * </p>
 *
 * @author Listens for BufferedImageStreamMotionEvents and assigns Point labels to BufferedImages based on the centre of gravity (COG)
 * of detected motion
 */
public class MotionCOGPosition3DLabelAssigner implements LabelAssigner<SerializableBufferedImageAdapter, Position3D>,
		BufferedImageStreamMotionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private BufferedImageStreamMotionDetector motionDetector;
	private Position3D motionCog; 
	private boolean sticky;
	private PointToPosition3DEstimator pointToPosition3DEstimator;
	
	/**
	 * <p>
	 * Constructor for MotionCOGPointLabelAssigner.
	 * </p>
	 *
	 * @param w
	 *            a int.
	 * @param h
	 *            a int.
	 * @param sticky
	 *            a boolean.
	 */
	public MotionCOGPosition3DLabelAssigner(int w, int h, boolean sticky, PointToPosition3DEstimator pointToPosition3DEstimator) {
		motionDetector = new BufferedImageStreamMotionDetector(w, h);
		motionDetector.addMotionListener(this);
		this.sticky = sticky;
		this.pointToPosition3DEstimator = pointToPosition3DEstimator;
	}

	/** {@inheritDoc} */
	@Override
	public LabeledData<SerializableBufferedImageAdapter, Position3D> assignLabel(SerializableBufferedImageAdapter data) {
		motionDetector.detect(data.getImage());
		Position3D point = motionCog;

		if (!sticky) {
			motionCog = null;
		}

		return new LabeledData<SerializableBufferedImageAdapter, Position3D>(data, point);

	}

	/** {@inheritDoc} */
	@Override
	public void motionDetected(BufferedImageStreamMotionEvent event) {
		motionCog = pointToPosition3DEstimator.getPosition3D(event.getCog());
	}

}
