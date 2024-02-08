package org.ml4j.tensor.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Assertions;
import org.ml4j.autograd.impl.AutogradValueProperties;
import org.ml4j.tensor.AutogradTestBase;
import org.ml4j.tensor.Size;


public class DJLAutogradWithoutNativeGradientTest extends AutogradTestBase<DJLTensor, DJLTensorOperations> {

    @Override
    protected DJLTensorImpl createGradValue(float value, boolean requires_grad) {
        return new DJLTensorImpl(() -> createData(value, size), new AutogradValueProperties<Size>().setContext(size).setRegistry(registry).setRequires_grad(requires_grad));
    }

    @Override
    protected DJLTensorImpl createGradValue(float value, boolean requires_grad, Size size) {
        return new DJLTensorImpl(() -> createData(value, size), new AutogradValueProperties<Size>().setContext(size).setRegistry(registry).setRequires_grad(requires_grad));
    }

    @Override
    protected DJLTensorImpl createRandomValue(boolean requires_grad, int... dims) {
        return createGradValue((float) Math.random(), requires_grad, new Size(dims));
    }

    @Override
    protected DJLTensorImpl createOnesValue(boolean requires_grad, int... dims) {
        return createGradValue(1, requires_grad, new Size(dims));
    }

    @Override
    protected DJLTensorImpl createGradValue(DJLTensorOperations value, boolean requires_grad) {
        return new DJLTensorImpl(() -> value, new AutogradValueProperties<Size>().setContext(size).setRegistry(registry).setRequires_grad(requires_grad));
    }

    private Shape getShape(Size size) {
        long[] dims = new long[size.dimensions().length];
        int ind = 0;
        for (int dim : size.dimensions()) {
            dims[ind] = dim;
            ind++;
        }
        return new Shape(dims);
    }

    @Override
    protected void assertEquals(DJLTensorOperations value1, DJLTensorOperations value2) {
        NDArray m1 = value1.getNDArray();
        NDArray m2 = value2.getNDArray();
        Assertions.assertEquals(m1.toFloatArray().length, m2.toFloatArray().length);
        for (int i = 0; i < m1.toFloatArray().length; i++) {
            Assertions.assertEquals(m1.toFloatArray()[i], m2.toFloatArray()[i], 0.01f);
        }
    }

    protected void assertArrayEqual(float[] actual, float[] expected, float delta) {
        Assertions.assertArrayEquals(expected, actual, delta);
    }

    @Override
    protected DJLTensorOperations add(DJLTensorOperations value1, DJLTensorOperations value2) {
        return value1.add(value2);
    }

    @Override
    protected DJLTensorOperations mul(DJLTensorOperations value1, float value2) {
        return value1.mul(value2);
    }


    @Override
    protected boolean isNativeGradientSupported() {
        return false;
        //return true;
    }

    @Override
    protected boolean isNativeGradientExpected() {
        return false;
    }

    @Override
    protected DJLTensorOperations createData(float value) {
        float[] data = new float[size.numel()];
        for (int i = 0; i < data.length; i++) {
            data[i] = value;
        }
        return new DJLTensorOperationsImpl(DJLTensorFactory.getManager().create(data, getShape(size)), getShape(size), false);
    }

    @Override
    protected DJLTensorOperations createData(float value, Size size) {
        float[] data = new float[size.numel()];
        for (int i = 0; i < data.length; i++) {
            data[i] = value;
        }
        return new DJLTensorOperationsImpl(DJLTensorFactory.getManager().create(data, getShape(size)), getShape(size), false);
    }
}
