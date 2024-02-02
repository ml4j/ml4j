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

import java.util.Date;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.ml4j.imaging.FrameUpdateListener;

/**
 * <p>
 * Logs the time delay detected between the timestamp of a source frame
 * (as determed by its id) and the time of the update notification received
 * </p>
 *
 * @author Michael Lavelle
 */
public class TimestampedIdDelayLogger<F> implements FrameUpdateListener<F, Long> {

	private long totalDelay;
	private int frameUpdates;
	
	/** Constant <code>LOGGER</code> */
	public static Logger LOGGER = Logger.getLogger(TimestampedIdDelayLogger.class);

	/** {@inheritDoc} */
	@Override
	public void onFrameUpdate(F frame, Long frameTimestamp) {

		int delay = (int) (new Date().getTime() - frameTimestamp.longValue());
		if (frameUpdates > 0) {
			int avg = (int) (totalDelay / frameUpdates);
			LOGGER.log(Level.INFO, "Average processing delay is:" + avg + " ms");
			
		}
		frameUpdates++;
		totalDelay = totalDelay + delay;
	}

}
