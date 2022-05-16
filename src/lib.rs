use numpy::ndarray::{Array, ArrayViewD, Ix, Ix2};
use numpy::{array, IntoPyArray, PyArray2, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
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
    ) -> &'py PyArray2<f64> {
        self.compute(py,x,y)
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

    // fn check<'py>(&self, //na obecny moment może być statyczna, potem może się to zmienić
    //                 py: Python<'py>,
    //                 x: PyReadonlyArrayDyn<f64>,
    // ) -> &'py PyResult<i32> {
    //     match x[0][0] {
    //         0 => Ok(x[0][0])
    //         _ => Err(EssentiaException)
    //     }
    //
    //     // z.into_pyarray(py)
    // }

    fn method<'py>(&self, py: Python<'py>, x: PyReadonlyArrayDyn<f64>) -> PyResult<f64> {
        let x = x.as_array();
        if (x.ndim()) == 2 {
            Err(EssentiaException::new_err("sth went wrong"))
        } else { Ok(x[[0,0]]) }
    }


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
    m.add_function(wrap_pyfunction!(divide, _py).unwrap())?;
    // m.add_function(wrap_pyfunction!(<ChromaCrossSimilarity as Trait>::divide, _py).unwrap())?;
    Ok(())
}

//
// use pyo3::exceptions::PyZeroDivisionError;
// use pyo3::prelude::*;
//
// #[pyfunction]
// fn divide(a: i32, b: i32) -> PyResult<i32> {
//     match a.checked_div(b) {
//         Some(q) => Ok(q),
//         None => Err(PyZeroDivisionError::new_err("division by zero")),
//     }
// }
//
// fn main(){
//     Python::with_gil(|py|{
//         let fun = pyo3::wrap_pyfunction!(divide, py).unwrap();
//         fun.call1((1,0)).unwrap_err();
//         fun.call1((1,1)).unwrap();
//     });
// }
