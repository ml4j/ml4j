package org.ml4j.imaging.tracking;

public interface VelocityFilter {

	public void positionUpdated(double position,long iteration);
	public Double getVelocityEstimate();
}
