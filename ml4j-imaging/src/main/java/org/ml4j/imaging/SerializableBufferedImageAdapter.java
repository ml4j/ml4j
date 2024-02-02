package org.ml4j.imaging;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

public class SerializableBufferedImageAdapter implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

    transient List<BufferedImage> images;
	
	public BufferedImage getImage() {
		return images.get(0);
	}
	
	public SerializableBufferedImageAdapter(BufferedImage image)
	{
		this.images = new ArrayList<BufferedImage>();
		this.images.add(image);
	}
	
	 private void writeObject(ObjectOutputStream out) throws IOException {
	        out.defaultWriteObject();
	        out.writeInt(images.size()); // how many images are serialized?
	        for (BufferedImage eachImage : images) {
	            ImageIO.write(eachImage, "png", out); // png is lossless
	        }
	    }

	    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
	        in.defaultReadObject();
	        final int imageCount = in.readInt();
	        images = new ArrayList<BufferedImage>(imageCount);
	        for (int i=0; i<imageCount; i++) {
	            images.add(ImageIO.read(in));
	        }
	    }
	
}
