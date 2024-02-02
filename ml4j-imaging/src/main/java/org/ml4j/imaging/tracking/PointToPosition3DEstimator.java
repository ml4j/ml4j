package org.ml4j.imaging.tracking;

import java.awt.Point;

public interface PointToPosition3DEstimator {
	
	public Position3D getPosition3D(Point point);

}
