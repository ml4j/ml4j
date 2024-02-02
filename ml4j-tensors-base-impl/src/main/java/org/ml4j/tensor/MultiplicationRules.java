package org.ml4j.tensor;

public class MultiplicationRules {

    public static Size[] matmul(Size first, Size second) {
        if (first.dimensions().length == 1 && second.dimensions().length == 1) {
            // If both tensors are 1-dimensional, the dot product (scalar) is returned.
            return new Size[] {first, second, new Size(1), new Size()};
        } else if (first.dimensions().length == 2 && second.dimensions().length == 2) {
            // If both arguments are 2-dimensional, the matrix-matrix product is returned.
            return new Size[] {first, second, new Size(first.dimensions()[0], second.dimensions()[1]), new Size(first.dimensions()[0], second.dimensions()[1])};
        } else if (first.dimensions().length == 1 && second.dimensions().length == 2) {
            // if the first argument is 1-dimensional and the second argument is 2-dimensional,
            // a 1 is prepended to its dimension for the purpose of the matrix multiply.
            // After the matrix multiply, the prepended dimension is removed.
            return new Size[] {new Size(1, first.dimensions()[0]), second, new Size(1, second.dimensions()[1]), new Size(second.dimensions()[1])};
        } else if (first.dimensions().length == 2 && second.dimensions().length == 1) {
            //If the first argument is 2-dimensional and the second argument is 1-dimensional, //
            // the matrix-vector product is returned.
            return new Size[] {first, second, new Size(first.dimensions()[1], 1), new Size(first.dimensions()[1], 1) };
        } else if (first.dimensions().length >= 1 && second.dimensions().length >= 1 && (first.dimensions().length > 2 || second.dimensions().length > 2)) {
            if (first.dimensions().length == 1) {
                //  If the first argument is 1-dimensional, a 1 is prepended to its dimension for the
                //  purpose of the batched matrix multiply and removed after.
                int[] dimsBefore = new int[second.dimensions().length];
                int[] dims = new int[second.dimensions().length - 1];
                dimsBefore[0] = 1;
                for (int i = 1; i < second.dimensions().length; i++) {
                    dims[i - 1] = second.dimensions()[i];
                    dimsBefore[i] = second.dimensions()[i];
                }
                return new Size[] {new Size(1, first.dimensions()[0]), second, new Size(dimsBefore), new Size(dims) };
            } else if (second.dimensions().length == 1) {
                //  If the second argument is 1-dimensional, a 1 is appended to its
                //  dimension for the purpose of the batched matrix multiple and removed after.
                int[] dims = new int[first.dimensions().length - 1];
                int[] dimsBefore = new int[first.dimensions().length];
                dimsBefore[dimsBefore.length - 1] = 1;
                for (int i = 0; i < first.dimensions().length - 1; i++) {
                    dims[i] = first.dimensions()[i];
                    dimsBefore[i] = first.dimensions()[i];
                }
                return new Size[] {first, new Size(second.dimensions()[0], 1), new Size(dimsBefore), new Size(dims)};
            } else {
                int[] firstPrefixes = new int[first.dimensions().length  - 2];
                int[] secondPrefixes = new int[second.dimensions().length  - 2];
                Size firstSuffix = new Size(first.dimensions()[first.dimensions().length - 2],
                        first.dimensions()[first.dimensions().length - 1]);
                Size secondSuffix = new Size(second.dimensions()[second.dimensions().length - 2],
                        second.dimensions()[second.dimensions().length - 1]);
                for (int i = 0; i < firstPrefixes.length; i++) {
                    firstPrefixes[i] = first.dimensions()[i];
                }
                for (int i = 0; i < secondPrefixes.length; i++) {
                    secondPrefixes[i] = second.dimensions()[i];
                }
                Size broadcast = getBroadcast(new Size(firstPrefixes), new Size(secondPrefixes));
                int numel = broadcast.numel();
                Size firstSuffix2 = multiplyFirstDim(firstSuffix, numel);
                Size[] outputBefore = matmul(firstSuffix2, secondSuffix);
                outputBefore[outputBefore.length - 1] = splitFirstDim(outputBefore[outputBefore.length - 1], broadcast.dimensions());
                return outputBefore;

            }
        } else {
            throw new IllegalArgumentException();
        }
    }

    private static Size multiplyFirstDim(Size firstSuffix, int numel) {
        int[] dims = firstSuffix.dimensions();
        dims[0] = dims[0] * numel;
        return new Size(dims);
    }

    private static Size splitFirstDim(Size firstSuffix, int[] numels) {
        int[] dims = firstSuffix.dimensions();
        int[] newDims = new int[dims.length + numels.length];
        int prod = 1;
        for (int i = 0; i < numels.length; i++) {
            newDims[i] = numels[i];
            prod = prod * numels[i];
        }
        newDims[numels.length] = dims[0] / prod;
        for (int i = 1; i < dims.length; i++) {
            newDims[i + numels.length] = dims[i];
        }
        return new Size(newDims);
    }

    public static Size getBroadcast(Size first, Size second) {

        if (first.dimensions().length >= 1 && second.dimensions().length >= 1) {
            int[] res = new int[Math.max(first.dimensions().length, second.dimensions().length)];
            for (int i = 0; i < Math.min(first.dimensions().length, second.dimensions().length); i++) {
                int f = first.dimensions()[first.dimensions().length - 1 - i];
                int s = second.dimensions()[second.dimensions().length - 1 - i];
                if (f == s || f == 1 || s == 1) {
                    int r = Math.max(f, s);
                    res[res.length - 1 - i] = r;
                } else {
                    throw new IllegalArgumentException();
                }
            }
            for (int i = 0; i < res.length - Math.min(first.dimensions().length, second.dimensions().length); i++) {
                if (first.dimensions().length > second.dimensions().length) {
                    res[i] = first.dimensions()[i];
                } else {
                    res[i] = second.dimensions()[i];
                }
            }
            Size s  = new Size(res);
            return s;
        } else {
            if (first.dimensions().length == 0) return second;
            if (second.dimensions().length == 0) return first;
            throw new IllegalArgumentException();
        }
    }

}