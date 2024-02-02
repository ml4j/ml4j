package org.ml4j.imaging.tracking;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class FIRVelocityFilter implements VelocityFilter, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int sampleIterationGap;
	private int samplesToConsider;
	private List<Double> positions;
	private List<Long> timeDiffs;
	private Long lastUpdateTime;
	private Double iterationDuration;


	public FIRVelocityFilter(int sampleIterationGap, int samplesToConsider) {
		this.sampleIterationGap = sampleIterationGap;
		this.samplesToConsider = samplesToConsider;
		this.positions = new ArrayList<Double>();
		this.timeDiffs = new ArrayList<Long>();
	}
	
	public FIRVelocityFilter(int sampleIterationGap, int samplesToConsider,double iterationDuration) {
		this.sampleIterationGap = sampleIterationGap;
		this.samplesToConsider = samplesToConsider;
		this.positions = new ArrayList<Double>();
		this.timeDiffs = new ArrayList<Long>();
		this.iterationDuration = iterationDuration;
	}

	@Override
	public void positionUpdated(double position, long iteration) {
		if (this.lastUpdateTime != null)
		{
			timeDiffs.add(new Date().getTime() - lastUpdateTime.longValue());
		}
		else
		{
			timeDiffs.add(0l);
		}
		this.lastUpdateTime = new Date().getTime();

		if (sampleIterationGap == 0
				|| (iteration - 1) % sampleIterationGap == 0) {
		
			positions.add(position);
			if (positions.size() > samplesToConsider)
			{
				positions.remove(0);
				timeDiffs.remove(0);
			}
		}
		
	}

	@Override
	public Double getVelocityEstimate() {

		int samples = Math.min(samplesToConsider, this.positions.size());
		if (samples < 2)
			return null;
		int startIndex = this.positions.size() - samples;
		if (startIndex < 0)
			startIndex = 0;
		double[] positionsArray = new double[samples];

		double totalTimeDiff = 0d;
		for (int i = 0; i < samples; i++) {
			positionsArray[i] = this.positions.get(startIndex);
			totalTimeDiff = totalTimeDiff + this.timeDiffs.get(startIndex);
			startIndex++;
		}
		double averageTime = iterationDuration == null ? totalTimeDiff/(1000 * samples) : iterationDuration.doubleValue();

		double[] weights = getWeights(positionsArray.length);
		double velocity = 0;
		for (int i = 0; i < positionsArray.length; i++) {
			velocity = velocity + weights[i] * positionsArray[i];
		}
		velocity = velocity / averageTime;

		return velocity;
	}

	private double[] getWeights(int samplesToConsider) {
		if (samplesToConsider == 2) {
			return new double[] { -1, 1 };
		} else if (samplesToConsider == 3) {
			return new double[] { -0.5, 0, 0.5 };
		} else if (samplesToConsider == 4) {
			return new double[] { -0.3, -0.1, 0.1, 0.3 };
		} else if (samplesToConsider == 5) {
			return new double[] { -0.2, -0.1, 0, 0.1, 0.2 };
		} else {
			throw new RuntimeException(
					"No weights specified for velocity calcuation for:"
							+ samplesToConsider + " samples");
		}
	}
}
