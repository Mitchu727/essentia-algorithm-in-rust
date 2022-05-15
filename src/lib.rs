use numpy::ndarray::{arr2, Array, ArrayD, ArrayView2, ArrayViewD, ArrayViewMutD, Ix2};
use numpy::{array, IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use pyo3::prelude::*;

#[pyclass(subclass)]
struct Algorithm {
    #[pyo3(get)]
    processingMode: usize,
    // processingMode: String,
}

#[pymethods]
impl Algorithm {
    #[new]
    fn new() -> Self {
        Algorithm { processingMode: 1 }
        // Algorithm { processingMode: String::from("Standard") }
    }

    pub fn method(&self) -> PyResult<usize> {
        Ok(self.processingMode)
    }
}

#[pyclass(extends=Algorithm, subclass)]
struct ChromaCrossSimilarity {
    #[pyo3(get)]
    otiBinary: bool,
    frameStackSize: i32,
}

#[pymethods]
impl ChromaCrossSimilarity {
    #[new]
    fn new(otiBinary: bool, frameStackSize: i32) -> (Self, Algorithm) {
        (ChromaCrossSimilarity{otiBinary, frameStackSize}, Algorithm::new())
    }

    // fn compute(&mut self, py_args: &PyTuple) -> PyResult<String> {
    //     Ok(format!(
    //         "py_args={:?}",
    //         py_args
    //     ))
    // }

    // wrapper of `chroma_cross_similarity`
    // #[pyo3(name = "compute")]
    fn compute<'py>(&self,
                       py: Python<'py>,
                       x: PyReadonlyArrayDyn<f64>,
                       y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray2<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = compute_internal(x,y);
        z.into_pyarray(py)
    }
}

fn compute_internal (x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> Array<f64, Ix2> {
    return array!([x[[1,0]] + y[[0,0]]])
}

//
// #[pyclass]
// #[derive(Debug, Clone)]
// struct ChromaCrossSimilarity {
//     #[pyo3(get)]
//     otiBinary: bool,
//     frameStackSize: i32,
// }
//
//
// #[pymethods]
// impl ChromaCrossSimilarity {
//     #[new]
//     fn new(otiBinary: bool, frameStackSize: i32,) -> Self {
//         ChromaCrossSimilarity{otiBinary, frameStackSize}
//     }
// }


#[pymodule]
fn essentia_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // example using immutable borrows producing a new array
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // fn chroma_cross_similarity(mut x: ArrayViewD<'_, f64>, mut y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    //     4.0 * &x
    // }

    // wrapper of `chroma_cross_similarity`
    // #[pyfn(m)]
    // #[pyo3(name = "ChromaCrossSimilarity")]
    // fn chroma_cross_similarity_py<'py>(
    //     py: Python<'py>,
    //     x: PyReadonlyArrayDyn<f64>,
    //     y: PyReadonlyArrayDyn<f64>,
    // ) -> &'py PyArrayDyn<f64> {
    //     let x = x.as_array();
    //     let y = y.as_array();
    //     let z = chroma_cross_similarity(x, y);
    //     z.into_pyarray(py)
    // }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }
    m.add_class::<ChromaCrossSimilarity>()?;

    Ok(())
}