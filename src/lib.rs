use numpy::ndarray::{arr2, Array, ArrayD, ArrayView2, ArrayViewD, ArrayViewMutD, Ix2};
use numpy::{array, IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use pyo3::prelude::*;

#[pyclass(subclass)]
struct Algorithm {
    #[pyo3(get)]
    processingMode: String,
}

#[pymethods]
impl Algorithm {
    #[new]
    fn new() -> Self {
        Algorithm { processingMode: String::from("Standard") }
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

    // wrapper of `compute_internal`
    fn compute<'py>(&self, //na obecny moment może być statyczna, potem może się to zmienić
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


#[pymodule]
fn essentia_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ChromaCrossSimilarity>()?;
    Ok(())
}