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
package org.ml4j.imaging.targets;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;

import javax.swing.ImageIcon;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.ml4j.imaging.FrameUpdateListener;
import org.ml4j.imaging.SerializableBufferedImageAdapter;

/**
 * <p>
 * A listener for updates of frames of type BufferedImage and id of type ID
 * which displays the frames on a JDialog
 * </p>
 *
 * @author Michael Lavelle
 */
public class ImageDisplay<ID> implements FrameUpdateListener<SerializableBufferedImageAdapter, ID> {

	private JLabel label;

	private JDialog f;

	public Point mouseClick;

	/**
	 * <p>
	 * close.
	 * </p>
	 */
	public void close() {
		f.dispatchEvent(new WindowEvent(f, WindowEvent.WINDOW_CLOSING));
		f.dispose();
	}

	/**
	 * <p>
	 * nextMouseClick.
	 * </p>
	 */
	public Point nextMouseClick() {
		mouseClick = null;
		while (mouseClick == null) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return mouseClick;

	}

	/** {@inheritDoc} */
	@Override
	public void finalize() throws Throwable {
		close();
		super.finalize();
	}

	/**
	 * <p>
	 * Constructor for ImageDisplay.
	 * </p>
	 *
	 * @param dimension
	 *            a {@link java.awt.Dimension} object.
	 */
	public ImageDisplay(Dimension dimension) {
		this((int) dimension.getWidth(), (int) dimension.getHeight());

	}

	/**
	 * <p>
	 * Constructor for ImageDisplay.
	 * </p>
	 *
	 * @param w
	 *            a int.
	 * @param h
	 *            a int.
	 */
	public ImageDisplay(int w, int h) {
		BufferedImage image = new BufferedImage(w, h, 1);
		label = new JLabel(new ImageIcon(image));
		f = new JDialog();
		f.setModalityType(JDialog.ModalityType.MODELESS);
		// f.setModal(true);
		f.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		f.getContentPane().add(label);
		f.pack();
		f.setSize(w, h);
		f.setLocation(0, 0);
		f.setVisible(true);

		label.addMouseListener(new java.awt.event.MouseAdapter() {
			public void mouseClicked(java.awt.event.MouseEvent evt) {
				ImageDisplay.this.mouseClick = evt.getPoint();
			}
		});

	}

	/** {@inheritDoc} */
	@Override
	public void onFrameUpdate(SerializableBufferedImageAdapter image, ID frameId) {
		// TODO Auto-generated method stub
		label.setIcon(new ImageIcon(image.getImage()));

	}

}
