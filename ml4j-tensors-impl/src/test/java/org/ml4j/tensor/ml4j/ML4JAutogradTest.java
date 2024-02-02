package org.ml4j.tensor.ml4j;

import org.junit.jupiter.api.Assertions;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.autograd.impl.AutogradValueProperties;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.tensor.AutogradTestBase;
import org.ml4j.tensor.Size;


public class ML4JAutogradTest extends AutogradTestBase<ML4JTensor, ML4JTensorOperations> {

    private static MatrixFactory matrixFactory = new JBlasRowMajorMatrixFactory();

    private static DirectedComponentsContext context = new DirectedComponentsContextImpl(matrixFactory, true);

    @Override
    protected ML4JTensorImpl createGradValue(float value, boolean requires_grad) {
        return new ML4JTensorImpl(context, () -> createData(value), new AutogradValueProperties<Size>().setContext(size).setRequires_grad(requires_grad));
    }

    @Override
    protected ML4JTensorImpl createGradValue(float value, boolean requires_grad, Size size) {
        return new ML4JTensorImpl(context, () -> createData(value, size), new AutogradValueProperties<Size>().setRegistry(registry).setContext(size).setRequires_grad(requires_grad));
    }

    @Override
    protected ML4JTensorImpl createGradValue(ML4JTensorOperations value, boolean requires_grad) {
        return new ML4JTensorImpl(context, () -> value, new AutogradValueProperties<Size>().setContext(size).setRegistry(registry).setRequires_grad(requires_grad));
    }

    @Override
    protected void assertEquals(ML4JTensorOperations value1, ML4JTensorOperations value2) {
        Matrix m1 = value1.getMatrix();
        Matrix m2 = value2.getMatrix();
        Assertions.assertEquals(m1.getLength(), m2.getLength());
        for (int i = 0; i < m1.getLength(); i++) {
            Assertions.assertEquals(m1.get(i), m2.get(i), 0.01f);
        }
    }

    protected void assertArrayEqual(float[] actual, float[] expected, float delta) {
        Assertions.assertArrayEquals(expected, actual, delta);
    }

    @Override
    protected ML4JTensorOperations add(ML4JTensorOperations value1, ML4JTensorOperations value2) {
        return value1.add(value2);
    }

    @Override
    protected ML4JTensorOperations mul(ML4JTensorOperations value1, float value2) {
        return value1.mul(value2);
    }

    @Override
    protected boolean isNativeGradientSupported() {
        return false;
    }

    @Override
    protected boolean isNativeGradientExpected() {
        return false;
    }

    @Override
    protected ML4JTensorOperations createData(float value) {
        return new ML4JTensorOperationsImpl(context, value, size);
    }

    @Override
    protected ML4JTensorOperations createData(float value, Size size) {
        return new ML4JTensorOperationsImpl(context, value, size);
    }

    @Override
    protected ML4JTensorImpl createRandomValue(boolean requires_grad, int... dims) {
        return createGradValue((float) Math.random(), requires_grad, new Size(dims));
    }

    @Override
    protected ML4JTensorImpl createOnesValue(boolean requires_grad, int... dims) {
        return createGradValue(1, requires_grad, new Size(dims));
    }

    @Override
    public void test_tensor_broadcast_addition() {

    }

    @Override
    public void test_tensor_broadcast_addition2() {

    }
}
