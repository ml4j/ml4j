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
package org.ml4j.imaging.tracking;

import java.util.Date;

import org.ml4j.imaging.FrameUpdateListener;
import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.mapping.LabeledData;

/**
 * <p>
 * MovingTargetPositionEstimator class.
 * </p>
 *
 * @author Michael Lavelle
 */
public class MovingTargetPositionEstimator implements FrameUpdateListener<LabeledData<SerializableBufferedImageAdapter, Position3D>, Long> {

	private Position3D estimatedPosition;
	private long estimationTime;
	private Velocity3D estimatedVelocity;
	private VelocityFilter leftRightVelocityFilter = new FIRVelocityFilter(0,5);
	private VelocityFilter upDownVelocityFilter= new FIRVelocityFilter(0,5);
	private VelocityFilter forwardBackVelocityFilter= new FIRVelocityFilter(0,5);

	public MovingTargetPositionEstimator(VelocityFilter leftRightVelocityFilter,VelocityFilter upDownVelocityFilter,VelocityFilter forwardBackVelocityFilter)
	{
		this.leftRightVelocityFilter = leftRightVelocityFilter;
		this.upDownVelocityFilter = upDownVelocityFilter;
		this.forwardBackVelocityFilter = forwardBackVelocityFilter;
	}
	
	/**
	 * <p>
	 * Getter for the field <code>estimatedPosition</code>.
	 * </p>
	 */
	public Position3D getEstimatedPosition() {
		return estimatedPosition;
	}

	/**
	 * <p>
	 * Getter for the field <code>estimationTime</code>.
	 * </p>
	 */
	public long getEstimationTime() {
		return estimationTime;
	}
	
	public Velocity3D getEstimatedVelocity()
	{
		return estimatedVelocity;
	}

	/**
	 * <p>
	 * getEstimationDelay.
	 * </p>
	 */
	public int getEstimationDelay() {
		return (int) (new Date().getTime() - estimationTime);
	}

	/** {@inheritDoc} */
	@Override
	public void onFrameUpdate(LabeledData<SerializableBufferedImageAdapter, Position3D> frame, Long frameId) {


		if (frame.getLabel() != null) {
			
			Position3D newPosition = frame.getLabel();
			long newEstimationTime = frameId.longValue();
			double time = (newEstimationTime - estimationTime)/1000d;
			if (estimatedPosition != null)
			{
				leftRightVelocityFilter.positionUpdated(newPosition.getLeftRightDist(), frameId);
				upDownVelocityFilter.positionUpdated(newPosition.getUpDownDist(), frameId);
				forwardBackVelocityFilter.positionUpdated(newPosition.getForwardDist(), frameId);

				
				if (leftRightVelocityFilter.getVelocityEstimate() != null && upDownVelocityFilter.getVelocityEstimate() != null && forwardBackVelocityFilter.getVelocityEstimate() != null && time != 0)
				{
					estimatedVelocity = new Velocity3D(leftRightVelocityFilter.getVelocityEstimate(),upDownVelocityFilter.getVelocityEstimate(),forwardBackVelocityFilter.getVelocityEstimate());
					
				}
				
			}
			else
			{
				estimatedVelocity = new Velocity3D(0,0,0);

			}
			
			estimatedPosition = newPosition;
		
			
			
			estimationTime = newEstimationTime;
		}

	}

}
