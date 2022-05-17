use numpy::ndarray::{Array, ArrayViewD, Ix2};
use numpy::{array, IntoPyArray, PyArray, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use pyo3::prelude::*;
use pyo3::create_exception;


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
    #[args(
    otiBinary = "true",
    frameStackSize = "1"
    )]
    fn new(otiBinary: bool, frameStackSize: i32) -> (Self, Algorithm) {
        (ChromaCrossSimilarity{otiBinary, frameStackSize}, Algorithm::new())
    }

    fn __call__<'py>(&self, //na obecny moment może być statyczna, potem może się to zmienić
                    py: Python<'py>,
                    x: PyReadonlyArrayDyn<f64>,
                    y: PyReadonlyArrayDyn<f64>,
    ) -> Result<&'py PyArray<f64, Ix2>, PyErr> {
        self.compute(py,x,y)
    }

    // wrapper of `compute_internal`
    fn compute<'py>(&self, //na obecny moment może być statyczna, potem może się to zmienić
                       py: Python<'py>,
                       x: PyReadonlyArrayDyn<f64>,
                       y: PyReadonlyArrayDyn<f64>,
    ) -> Result<&'py PyArray<f64, Ix2>, PyErr> {
        if x.ndim() != 2 || y.ndim() != 2 {
            return Err(EssentiaException::new_err("Wrong array dimensions"))
        }
        let x = x.as_array();
        let y = y.as_array();
        let z = self.compute_internal(x,y);
        Ok(z.into_pyarray(py))
    }

}

impl ChromaCrossSimilarity{
    fn compute_internal (&self, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> Array<f64, Ix2> {
        return array!([x[[1,0]] + y[[0,0]]])
    }
}

create_exception!(essentia_rust, EssentiaException, pyo3::exceptions::PyException);


#[pymodule]
fn essentia_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ChromaCrossSimilarity>()?;
    m.add("EssentiaException", _py.get_type::<EssentiaException>())?;
    Ok(())
}

// Tests -------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test() {
        let test_object = ChromaCrossSimilarity{
            otiBinary: true,
            frameStackSize: 1,
        };
        let x = array!([0.,1.],[2.,3.]);
        let y = array!([4.,5.],[6.,7.]);
        let outcome = test_object.compute_internal(x.view().into_dyn(),y.view().into_dyn());
        assert_eq!(array!([6.0]), outcome)
    }

}