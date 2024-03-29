package org.ml4j.imaging.tracking.sarxos;

import java.awt.Point;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.sarxos.webcam.WebcamExceptionHandler;
import com.github.sarxos.webcam.WebcamMotionListener;
import com.github.sarxos.webcam.util.jh.JHBlurFilter;
import com.github.sarxos.webcam.util.jh.JHGrayFilter;

/**
 * Webcam motion detector.
 *
 * Cloned from https://github.com/sarxos/webcam-capture/blob/master/webcam-capture/src/main/java/com/github/sarxos/webcam/WebcamMotionDetector.java
 * and modified by Michael Lavelle to operate on BufferedImage streams
 *
 * @author Bartosz Firyn (SarXos)
 */
public class BufferedImageStreamMotionDetector {

	/**
	 * Logger.
	 */
	private static final Logger LOG = LoggerFactory.getLogger(BufferedImageStreamMotionDetector.class);

	/**
	 * Default pixel difference intensity threshold (set to 25).
	 */
	public static final int DEFAULT_PIXEL_THREASHOLD = 25;

	/**
	 * Default check interval (in milliseconds, set to 1 second).
	 */
	public static final int DEFAULT_INTERVAL = 1000;

	/**
	 * Default percentage image area fraction threshold (set to 0.2%).
	 */
	public static final double DEFAULT_AREA_THREASHOLD = 0.2;

	/**
	 * <p>
	 * handleImage.
	 * </p>
	 *
	 * @param bufferedImage
	 *            a {@link java.awt.image.BufferedImage} object.
	 */
	public void handleImage(BufferedImage bufferedImage) {
		detect(bufferedImage);

		int delay = inertia != -1 ? inertia : 2 * interval;

		if (lastMotionTimestamp + delay < System.currentTimeMillis()) {
			motion.set(false);
		}

	}

	/**
	 * Motion listeners.
	 */
	private final List<BufferedImageStreamMotionListener> listeners = new ArrayList<BufferedImageStreamMotionListener>();

	/**
	 * Is detector running?
	 */
	private final AtomicBoolean running = new AtomicBoolean(false);

	/**
	 * Is motion?
	 */
	private final AtomicBoolean motion = new AtomicBoolean(false);

	/**
	 * Previously captured image.
	 */
	private BufferedImage previous = null;

	/**
	 * Motion check interval (1000 ms by default).
	 */
	private volatile int interval = DEFAULT_INTERVAL;

	/**
	 * Pixel intensity threshold (0 - 255).
	 */
	private volatile int pixelThreshold = DEFAULT_PIXEL_THREASHOLD;

	/**
	 * Pixel intensity threshold (0 - 100).
	 */
	private volatile double areaThreshold = DEFAULT_AREA_THREASHOLD;

	/**
	 * How long motion is valid (in milliseconds). Default value is 2 seconds.
	 */
	private volatile int inertia = -1;

	/**
	 * Motion strength (0 = no motion, 100 = full image covered by motion).
	 */
	private double area = 0;

	/**
	 * Center of motion gravity.
	 */
	private Point cog = null;

	/**
	 * Timestamp when motion has been observed last time.
	 */
	private volatile long lastMotionTimestamp = 0;

	/**
	 * Blur filter instance.
	 */
	private final JHBlurFilter blur = new JHBlurFilter(6, 6, 1);

	/**
	 * Gray filter instance.
	 */
	private final JHGrayFilter gray = new JHGrayFilter();

	/**
	 * Create motion detector. Will open webcam if it is closed.
	 *
	 * @param pixelThreshold
	 *            intensity threshold (0 - 255)
	 * @param areaThreshold
	 *            percentage threshold of image covered by motion
	 * @param interval
	 *            the check interval
	 * @param w
	 *            a int.
	 * @param h
	 *            a int.
	 */
	public BufferedImageStreamMotionDetector(int w, int h, int pixelThreshold, double areaThreshold, int interval) {

		setPixelThreshold(pixelThreshold);
		setAreaThreshold(areaThreshold);
		setInterval(interval);

		cog = new Point(w / 2, h / 2);
	}

	/**
	 * Create motion detector with default parameter inertia = 0.
	 *
	 * @param pixelThreshold
	 *            intensity threshold (0 - 255)
	 * @param w
	 *            a int.
	 * @param h
	 *            a int.
	 * @param areaThreshold
	 *            a double.
	 */
	public BufferedImageStreamMotionDetector(int w, int h, int pixelThreshold, double areaThreshold) {
		this(w, h, pixelThreshold, areaThreshold, DEFAULT_INTERVAL);
	}

	/**
	 * Create motion detector with default parameter inertia = 0.
	 *
	 * @param pixelThreshold
	 *            intensity threshold (0 - 255)
	 * @param w
	 *            a int.
	 * @param h
	 *            a int.
	 */
	public BufferedImageStreamMotionDetector(int w, int h, int pixelThreshold) {
		this(w, h, pixelThreshold, DEFAULT_AREA_THREASHOLD);
	}

	/**
	 * Create motion detector with default parameters - threshold = 25, inertia
	 * = 0.
	 *
	 * @param w
	 *            a int.
	 * @param h
	 *            a int.
	 */
	public BufferedImageStreamMotionDetector(int w, int h) {
		this(w, h, DEFAULT_PIXEL_THREASHOLD);
	}

	/**
	 * <p>
	 * detect.
	 * </p>
	 *
	 * @param current
	 *            a {@link java.awt.image.BufferedImage} object.
	 */
	public void detect(BufferedImage current) {

		current = blur.filter(current, null);
		current = gray.filter(current, null);

		int p = 0;

		int cogX = 0;
		int cogY = 0;

		int w = current.getWidth();
		int h = current.getHeight();

		if (previous != null) {
			for (int x = 0; x < w; x++) {
				for (int y = 0; y < h; y++) {

					int cpx = current.getRGB(x, y);
					int ppx = previous.getRGB(x, y);
					int pid = combinePixels(cpx, ppx) & 0x000000ff;

					if (pid >= pixelThreshold) {
						cogX += x;
						cogY += y;
						p += 1;
					}
				}
			}
		}

		area = p * 100d / (w * h);

		if (area >= areaThreshold) {

			cog = new Point(cogX / p, cogY / p);

			if (motion.compareAndSet(false, true)) {
				lastMotionTimestamp = System.currentTimeMillis();
			}

			notifyMotionListeners();

		} else {
			cog = new Point(w / 2, h / 2);
		}

		previous = current;
	}

	/**
	 * Will notify all attached motion listeners.
	 */
	private void notifyMotionListeners() {
		BufferedImageStreamMotionEvent wme = new BufferedImageStreamMotionEvent(this, area, cog);
		for (BufferedImageStreamMotionListener l : listeners) {
			try {
				l.motionDetected(wme);
			} catch (Exception e) {
				WebcamExceptionHandler.handle(e);
			}
		}
	}

	/**
	 * Add motion listener.
	 *
	 * @param l
	 *            listener to add
	 * @return true if listeners list has been changed, false otherwise
	 */
	public boolean addMotionListener(BufferedImageStreamMotionListener l) {
		return listeners.add(l);
	}

	/**
	 * <p>
	 * getMotionListeners.
	 * </p>
	 *
	 * @return All motion listeners as array
	 */
	public WebcamMotionListener[] getMotionListeners() {
		return listeners.toArray(new WebcamMotionListener[listeners.size()]);
	}

	/**
	 * Removes motion listener.
	 *
	 * @param l
	 *            motion listener to remove
	 * @return true if listener was available on the list, false otherwise
	 */
	public boolean removeMotionListener(WebcamMotionListener l) {
		return listeners.remove(l);
	}

	/**
	 * <p>
	 * Getter for the field <code>interval</code>.
	 * </p>
	 *
	 * @return Motion check interval in milliseconds
	 */
	public int getInterval() {
		return interval;
	}

	/**
	 * Motion check interval in milliseconds. After motion is detected, it's
	 * valid for time which is equal to value of 2 * interval.
	 *
	 * @param interval
	 *            the new motion check interval (ms)
	 * @see #DEFAULT_INTERVAL
	 */
	public void setInterval(int interval) {

		if (interval < 100) {
			throw new IllegalArgumentException("Motion check interval cannot be less than 100 ms");
		}

		this.interval = interval;
	}

	/**
	 * Set pixel intensity difference threshold above which pixel is classified
	 * as "moved". Minimum value is 0 and maximum is 255. Default value is 10.
	 * This value is equal for all RGB components difference.
	 *
	 * @param threshold
	 *            the pixel intensity difference threshold
	 * @see #DEFAULT_PIXEL_THREASHOLD
	 */
	public void setPixelThreshold(int threshold) {
		if (threshold < 0) {
			throw new IllegalArgumentException("Pixel intensity threshold cannot be negative!");
		}
		if (threshold > 255) {
			throw new IllegalArgumentException("Pixel intensity threshold cannot be higher than 255!");
		}
		this.pixelThreshold = threshold;
	}

	/**
	 * Set percentage fraction of detected motion area threshold above which it
	 * is classified as "moved". Minimum value for this is 0 and maximum is 100,
	 * which corresponds to full image covered by spontaneous motion.
	 *
	 * @param threshold
	 *            the percentage fraction of image area
	 * @see #DEFAULT_AREA_THREASHOLD
	 */
	public void setAreaThreshold(double threshold) {
		if (threshold < 0) {
			throw new IllegalArgumentException("Area fraction threshold cannot be negative!");
		}
		if (threshold > 100) {
			throw new IllegalArgumentException("Area fraction threshold cannot be higher than 100!");
		}
		this.areaThreshold = threshold;
	}

	/**
	 * Set motion inertia (time when motion is valid). If no value specified
	 * this is set to 2 * interval. To reset to default value,
	 * {@link #clearInertia()} method must be used.
	 *
	 * @param inertia
	 *            the motion inertia time in milliseconds
	 * @see #clearInertia()
	 */
	public void setInertia(int inertia) {
		if (inertia < 0) {
			throw new IllegalArgumentException("Inertia time must not be negative!");
		}
		this.inertia = inertia;
	}

	/**
	 * Reset inertia time to value calculated automatically on the base of
	 * interval. This value will be set to 2 * interval.
	 */
	public void clearInertia() {
		this.inertia = -1;
	}

	/**
	 * <p>
	 * isMotion.
	 * </p>
	 */
	public boolean isMotion() {
		if (!running.get()) {
			LOG.warn("Motion cannot be detected when detector is not running!");
		}
		return motion.get();
	}

	/**
	 * Get percentage fraction of image covered by motion. 0 means no motion on
	 * image and 100 means full image covered by spontaneous motion.
	 *
	 * @return Return percentage image fraction covered by motion
	 */
	public double getMotionArea() {
		return area;
	}

	/**
	 * Get motion center of gravity. When no motion is detected this value
	 * points to the image center.
	 *
	 * @return Center of gravity point
	 */
	public Point getMotionCog() {
		return cog;
	}

	/**
	 * <p>
	 * combinePixels.
	 * </p>
	 *
	 * @param rgb1
	 *            a int.
	 * @param rgb2
	 *            a int.
	 */
	private static int combinePixels(int rgb1, int rgb2) {

		// first ARGB

		int a1 = (rgb1 >> 24) & 0xff;
		int r1 = (rgb1 >> 16) & 0xff;
		int g1 = (rgb1 >> 8) & 0xff;
		int b1 = rgb1 & 0xff;

		// second ARGB

		int a2 = (rgb2 >> 24) & 0xff;
		int r2 = (rgb2 >> 16) & 0xff;
		int g2 = (rgb2 >> 8) & 0xff;
		int b2 = rgb2 & 0xff;

		r1 = clamp(Math.abs(r1 - r2));
		g1 = clamp(Math.abs(g1 - g2));
		b1 = clamp(Math.abs(b1 - b2));

		// in case if alpha is enabled (translucent image)

		if (a1 != 0xff) {
			a1 = a1 * 0xff / 255;
			int a3 = (255 - a1) * a2 / 255;
			r1 = clamp((r1 * a1 + r2 * a3) / 255);
			g1 = clamp((g1 * a1 + g2 * a3) / 255);
			b1 = clamp((b1 * a1 + b2 * a3) / 255);
			a1 = clamp(a1 + a3);
		}

		return (a1 << 24) | (r1 << 16) | (g1 << 8) | b1;
	}

	/**
	 * Clamp a value to the range 0..255
	 *
	 * @param c
	 *            a int.
	 */
	private static int clamp(int c) {
		if (c < 0) {
			return 0;
		}
		if (c > 255) {
			return 255;
		}
		return c;
	}

}
