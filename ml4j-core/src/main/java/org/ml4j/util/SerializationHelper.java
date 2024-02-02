package org.ml4j.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamClass;
import java.io.Serializable;
import java.net.URL;

public class SerializationHelper {
	private String path;
	private ClassLoader classLoader;

	public SerializationHelper(String path) {
		this.path = path;
	}

	public SerializationHelper(ClassLoader classLoader, String path) {
		this.classLoader = classLoader;
		this.path = path;
	}

	@SuppressWarnings("unchecked")
	public <S extends Serializable> S deserialize(Class<S> clazz, String id) {

		try {

			long uid = ObjectStreamClass.lookup(clazz).getSerialVersionUID();

			InputStream is = null;
			if (classLoader == null) {
				is = new FileInputStream(path + "/" + clazz.getName() + "/"
						+ uid + "/" + id + ".ser");
			} else {
				is = classLoader.getResourceAsStream(path + "/"
						+ clazz.getName() + "/" + uid + "/" + id + ".ser");
			}

			ObjectInputStream in = new ObjectInputStream(is);
			S e = (S) in.readObject();
			in.close();
			is.close();
			return e;

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public void serialize(Serializable e, String id) {

		try {
			Class<?> cl = e.getClass();
			long uid = ObjectStreamClass.lookup(cl).getSerialVersionUID();

			String classDir = path + "/" + e.getClass().getName();
			String dir = classDir + "/" + uid;

			URL baseUrl = classLoader != null ? classLoader.getResource(path)
					: null;
			File baseDir = classLoader != null ? new File(baseUrl.toURI()
					.getPath()) : null;

			FileOutputStream fileOut = null;
			File classDirFile = null;
			if (classLoader != null) {
				URL classUrl = classLoader.getResource(classDir);
				if (classUrl != null) {
					classDirFile = new File(classUrl.toURI().getPath());
					if (!classDirFile.exists()) {
						classDirFile.mkdir();
					}
				} else {
					classDirFile = new File(baseDir, e.getClass().getName());

					if (!classDirFile.exists()) {
						classDirFile.mkdir();
						System.out.println("Created:" + classDirFile);

					}
				}

				URL dirUrl = classLoader.getResource(dir);
				File dirFile = null;
				if (dirUrl != null) {

					dirFile = new File(dirUrl.toURI().getPath());
					if (!dirFile.exists()) {
						dirFile.mkdir();
					}

				} else {
					dirFile = new File(classDirFile, "" + uid);
					if (!dirFile.exists()) {
						dirFile.mkdir();
					}

				}

				URL fileUrl = classLoader.getResource(dir + "/" + id + ".ser");

				if (fileUrl != null) {

					File f = new File(fileUrl.toURI().getPath());

					fileOut = new FileOutputStream(f);

				} else {
					File f = new File(dirFile, id + ".ser");

					fileOut = new FileOutputStream(f);
				}

			} else {

				classDirFile = new File(classDir);
				if (!classDirFile.exists()) {
					classDirFile.mkdir();
				}

				File dirFile = new File(dir);
				if (!dirFile.exists()) {
					dirFile.mkdir();
				}

				fileOut = new FileOutputStream(dir + "/" + id + ".ser");

			}

			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(e);
			out.close();
			fileOut.close();

		} catch (Exception e1) {
			throw new RuntimeException(e1);
		}

	}

}
