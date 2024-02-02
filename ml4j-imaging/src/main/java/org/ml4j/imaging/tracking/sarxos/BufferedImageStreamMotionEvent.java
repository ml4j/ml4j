package org.ml4j.imaging.tracking.sarxos;

import java.awt.Point;
import java.util.EventObject;

/**
 * Buffered image stream detected motion event.
 *
 * Cloned from https://github.com/sarxos/webcam-capture/blob/master/webcam-capture/src/main/java/com/github/sarxos/webcam/WebcamMotionEvent.java
 * and modified by Michael Lavelle to relate to BufferedImage stream motion events
 *
 * @author Bartosz Firyn (SarXos)
 */
public class BufferedImageStreamMotionEvent extends EventObject {

	/** Constant <code>serialVersionUID=-7245768099221999443L</code> */
	private static final long serialVersionUID = -7245768099221999443L;

	private final double strength;
	private final Point cog;

	/**
	 * Create detected motion event.
	 *
	 * @param detector
	 *            a
	 *            {@link org.ml4j.imaging.tracking.sarxos.BufferedImageStreamMotionDetector}
	 *            object.
	 * @param strength
	 *            a double.
	 * @param cog
	 *            a {@link java.awt.Point} object.
	 */
	public BufferedImageStreamMotionEvent(BufferedImageStreamMotionDetector detector, double strength, Point cog) {

		super(detector);

		this.strength = strength;
		this.cog = cog;
	}

	/**
	 * Get percentage fraction of image covered by motion. 0 is no motion on
	 * image, and 100 is full image covered by motion.
	 *
	 * @return Motion area
	 */
	public double getArea() {
		return strength;
	}

	/**
	 * <p>
	 * Getter for the field <code>cog</code>.
	 * </p>
	 */
	public Point getCog() {
		return cog;
	}
}
