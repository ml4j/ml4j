package org.ml4j.imaging.tracking;

import java.awt.Point;

public class PointAsPosition3DEstimator implements PointToPosition3DEstimator {

	@Override
	public Position3D getPosition3D(Point point) {
		return new Position3D(point.getX(),point.getY(),0,point);
	}

}
