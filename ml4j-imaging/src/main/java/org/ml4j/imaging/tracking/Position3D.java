package org.ml4j.imaging.tracking;

import java.awt.Point;
import java.io.Serializable;

public class Position3D implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private double leftRightDist;
	private double upDownDist;
	private double forwardDist;
	private Point pointOnCamera;
	
	public Position3D(double leftRightDist,double upDownDist,double forwardDist,Point pointOnCamera)
	{
		this.leftRightDist = leftRightDist;
		this.upDownDist = upDownDist;
		this.forwardDist = forwardDist;
		this.pointOnCamera = pointOnCamera;
	} 
	
	public String toString()
	{
		return leftRightDist + "," + upDownDist + "," + forwardDist;
	}
	
	public Point getPointOnCamera() {
		return pointOnCamera;
	}



	public double getLeftRightDist() {
		return leftRightDist;
	}
	public void setLeftRightDist(double leftRightDist) {
		this.leftRightDist = leftRightDist;
	}
	public double getUpDownDist() {
		return upDownDist;
	}
	public void setUpDownDist(double upDownDist) {
		this.upDownDist = upDownDist;
	}
	public double getForwardDist() {
		return forwardDist;
	}
	public void setForwardDist(double forwardDist) {
		this.forwardDist = forwardDist;
	}
	
	
	
}
