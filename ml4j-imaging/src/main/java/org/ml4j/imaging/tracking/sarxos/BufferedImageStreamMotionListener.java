package org.ml4j.imaging.tracking.sarxos;

/**
 * Buffered Image Stream motion listener used to signal motion detection.
 *
 * Cloned from https://github.com/sarxos/webcam-capture/blob/master/webcam-capture/src/main/java/com/github/sarxos/webcam/WebcamDiscoveryListener.java
 * and modified by Michael Lavelle to listen to BufferedImageStreamMotionEvents
 *
 * @author bartosz Firyn (SarXos)
 */
public interface BufferedImageStreamMotionListener {

	/**
	 * Will be called after motion is detected.
	 *
	 * @param wme
	 *            motion event
	 */
	public void motionDetected(BufferedImageStreamMotionEvent wme);

}
