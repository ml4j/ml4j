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

/**
 * <p>
 * FilenameFromTimestampGenerator class.
 * </p>
 *
 * @author Michael Lavelle
 */
public class FilenameFromTimestampGenerator implements LabelFromIdExtractor<Long, String> {

	private String extension;

	/**
	 * <p>
	 * Constructor for FilenameFromTimestampGenerator.
	 * </p>
	 *
	 * @param extension
	 *            a {@link java.lang.String} object.
	 */
	public FilenameFromTimestampGenerator(String extension) {
		this.extension = extension;
	}

	/** {@inheritDoc} */
	@Override
	public String getLabelFromId(Long id) {
		return id.toString() + "." + extension;
	}

}
