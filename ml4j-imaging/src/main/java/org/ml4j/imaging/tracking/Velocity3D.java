package org.ml4j.imaging.tracking;


public class Velocity3D {

	private double leftRightVel;
	private double upDownVel;
	private double forwardVel;
	
	public Velocity3D(double leftRightVel,double upDownVel,double forwardVel)
	{
		this.leftRightVel = leftRightVel;
		this.upDownVel = upDownVel;
		this.forwardVel = forwardVel;
	}
	
	public String toString()
	{
		return leftRightVel + "," + upDownVel + "," + forwardVel;
	}

	public double getLeftRightVel() {
		return leftRightVel;
	}

	public void setLeftRightVel(double leftRightVel) {
		this.leftRightVel = leftRightVel;
	}

	public double getUpDownVel() {
		return upDownVel;
	}

	public void setUpDownVel(double upDownVel) {
		this.upDownVel = upDownVel;
	}

	public double getForwardVel() {
		return forwardVel;
	}

	public void setForwardVel(double forwardVel) {
		this.forwardVel = forwardVel;
	} 
	
	

	
	
}
