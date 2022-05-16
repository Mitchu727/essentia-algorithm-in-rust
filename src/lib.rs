use numpy::ndarray::{Array, ArrayBase, ArrayViewD, Dim, Ix2, OwnedRepr};
use numpy::{array, IntoPyArray, PyArray, PyArray2, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use pyo3::callback::IntoPyCallbackOutput;
use pyo3::prelude::*;
use pyo3::create_exception;
use pyo3::impl_::pyfunction::wrap_pyfunction;


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
        let z = compute_internal(x,y);
        Ok(z.into_pyarray(py))
    }

    // fn check_dimensions<'py>(&self,
    //                      py: Python<'py>,
    //                      x: PyReadonlyArrayDyn<f64>,
    //                      y: PyReadonlyArrayDyn<f64>,
    // ) -> Result<&'py PyArray<f64, Ix2>, PyErr> {
    //     let x = x.as_array();
    //     if x.ndim() != 2 || y.ndim() != 2 {
    //         Err(EssentiaException::new_err("sth went wrong"))
    //     } else { Ok(()) }
    // }

    // fn method<'py>(&self, py: Python<'py>, x: PyReadonlyArrayDyn<f64>) -> &'py PyArray2<f64> {
    //     let x = x.as_array();
    //     if (x.ndim()) == 2 {
    //         EssentiaException::new_err("sth went wrong").restore(py);
    //     };
    //     array!([x[[0,0]]]).into_pyarray(py)
    // }

}


#[pyfunction]
fn divide(a: f64, b: f64) -> PyResult<f64> {
    // match a.checked_div(b) {
    //     Some(q) => Ok(q),
    //     None => Err(EssentiaException::new_err("division by zero")),
    // }
    match a {
        0. => Ok(a),
        _ => Err(EssentiaException::new_err("division by zero")),
    }
}

fn compute_internal (x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> Array<f64, Ix2> {
    return array!([x[[1,0]] + y[[0,0]]])
}

create_exception!(essentia_rust, EssentiaException, pyo3::exceptions::PyException);


#[pymodule]
fn essentia_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ChromaCrossSimilarity>()?;
    m.add("EssentiaException", _py.get_type::<EssentiaException>())?;
    Ok(())
}