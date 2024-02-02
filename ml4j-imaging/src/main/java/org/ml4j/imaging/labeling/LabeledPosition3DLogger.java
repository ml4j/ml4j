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

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.ml4j.imaging.FrameUpdateListener;
import org.ml4j.imaging.tracking.Position3D;
import org.ml4j.mapping.LabeledData;

/**
 * <p>
 * Logs the time delay detected between the timestamp of a source frame
 * (as determed by its id) and the time of the update notification received
 * </p>
 *
 * @author Michael Lavelle
 */
public class LabeledPosition3DLogger<F extends Serializable,ID> implements FrameUpdateListener<LabeledData<F,Position3D>, ID> {


	
	/** Constant <code>LOGGER</code> */
	public static Logger LOGGER = Logger.getLogger(LabeledPosition3DLogger.class);

	/** {@inheritDoc} */
	@Override
	public void onFrameUpdate(LabeledData<F,Position3D> frame, ID frameId) {

		if (frame.getLabel() != null)
		{
			LOGGER.log(Level.INFO, "Labeled Point is:" + frame.getLabel().getPointOnCamera().getX() + "," + frame.getLabel().getPointOnCamera().getY());
		}
	}

}
