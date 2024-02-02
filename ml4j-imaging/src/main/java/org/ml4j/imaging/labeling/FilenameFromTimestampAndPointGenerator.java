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
package org.ml4j.imaging.labeling;

import java.awt.Point;

/**
 * <p>
 * FilenameFromTimestampAndPointGenerator class.
 * </p>
 *
 * @author Michael Lavelle
 */
public class FilenameFromTimestampAndPointGenerator implements LabelFromIdAndLabelExtractor<Long, Point, String> {

	private String extension;

	/**
	 * <p>
	 * Constructor for FilenameFromTimestampAndPointGenerator.
	 * </p>
	 *
	 * @param extension
	 *            a {@link java.lang.String} object.
	 */
	public FilenameFromTimestampAndPointGenerator(String extension) {
		this.extension = extension;
	}

	/** {@inheritDoc} */
	@Override
	public String getLabelFromIdAndLabel(Long id, Point label) {
		return id.toString() + "_" + ((int) label.getX()) + "_" + ((int) label.getY()) + "." + extension;
	}

}
