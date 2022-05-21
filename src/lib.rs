use numpy::ndarray::{Array, Array2, ArrayViewD, Axis, Ix, Ix2, s, ShapeBuilder};
use numpy::{array, IntoPyArray, PyArray, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use pyo3::prelude::*;
use pyo3::create_exception;
use rulinalg::utils::{dot, argmax};


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
    frameStackSize: usize,
    frame_stack_stride: usize,
    noti: u32,

}

#[pymethods]
impl ChromaCrossSimilarity {
    #[new]
    #[args(
    otiBinary = "true",
    frameStackSize = "9",
    )]
    fn new(otiBinary: bool, frameStackSize: usize) -> (Self, Algorithm) {
        (ChromaCrossSimilarity{otiBinary, frameStackSize, frame_stack_stride: 1, noti: 12}, Algorithm::new())
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
    fn compute_internal (&self, query_feature: ArrayViewD<'_, f64>, reference_feature: ArrayViewD<'_, f64>) -> Array<f64, Ix2> {
        let mathc_coef = 1.;
        let mismatch_coef = 0.;
        if self.otiBinary {
            let stack_frames_a = stack_chroma_frames(from_ndarray_to_vecs(query_feature), self.frameStackSize, self.frame_stack_stride);
            let stack_frames_b = stack_chroma_frames(from_ndarray_to_vecs(reference_feature), self.frameStackSize, self.frame_stack_stride);
            // print!("{:?}", stack_frames_a[0]);
            // print!("{:?}", stack_frames_a[1]);
            return chroma_cross_binary_sim_matrix(stack_frames_a, stack_frames_b, self.noti, mathc_coef, mismatch_coef)
            // return from_vecs_to_ndarray(from_ndarray_to_vecs(query_feature))
        }
        // else {  }
        return array!([query_feature[[1,0]] + reference_feature[[0,0]]])

    }
}

fn chroma_cross_binary_sim_matrix(chroma_a: Vec<Vec<f64>>, chroma_b: Vec<Vec<f64>>, n_shifts: u32, match_coef:f64, mismatch_coef: f64) -> Array2<f64> {
    let mut value_at_shifts = Vec::new();
    let mut oti_index;
    let mut sim_matrix = Array2::default([chroma_a.len(), chroma_b.len()]);
    for i in 0..chroma_a.len() {
        for j in 0..chroma_b.len() {
            for k in 0..n_shifts {
                let mut chroma_b_copy = chroma_b[j].to_vec();
                chroma_b_copy.rotate_right(k as usize);
                // value_at_shifts.push(chroma_a.slice(s![..,i]).dot(&chroma_b_copy));
                value_at_shifts.push(dot(&chroma_b_copy, &chroma_a[i]));
            }
            oti_index = argmax(&value_at_shifts);
            // {
            //     Ok(value) => oti_index = value,
            //     Err(_) => panic!("array in chroma_cross_binary_sim_matrix is empty"),
            // };
            value_at_shifts.clear();
            if oti_index.0 == 0 || oti_index.0 == 1 {
                 sim_matrix[[i,j]] = match_coef
            }
            else {
                sim_matrix[[i,j]] = mismatch_coef
            }
        }
    }
    return sim_matrix
}

fn stack_chroma_frames(frames: Vec<Vec<f64>>, frame_stack_size: usize, frame_stack_stride: usize) -> Vec<Vec<f64>> {
    if frame_stack_size == 1 {
        return frames;
    }
    let mut stop_idx: usize;
    let increment = frame_stack_size + frame_stack_stride;

    // if frames.len() < increment + 1 {
    //     return Err(EssentiaException::new_err("Wrong array dimensions"))
    // }
    // let mut stacked_frames: Array2<f64> = Array::zeros((frames.len() - increment, frames.shape[0] * frameStackSize));

    let mut stacked_frames = Vec::new();
    let mut stack = Vec::new();
    for i in (0..(frames.len() - increment)).step_by(frame_stack_stride) {
        stop_idx = i + increment;
        for start_time in (i..stop_idx).step_by(frame_stack_stride) {
            let sth = frames[start_time].to_vec();
            stack.extend_from_slice(&sth)
        }
        stacked_frames.push(stack.to_vec());
        stack.clear()
    }
    return stacked_frames;
}

    fn from_vecs_to_ndarray(vec_array :Vec<Vec<f64>>) -> Array2<f64> {
        let mut array = Array2::<f64>::default((vec_array.len(), vec_array[0].len()));
        for (i, mut row) in array.axis_iter_mut(Axis(0)).enumerate() {
            for (j, col) in row.iter_mut().enumerate() {
                *col = vec_array[i][j];
            }
        }
        return array
    }

    fn from_ndarray_to_vecs(array: ArrayViewD<f64>) -> Vec<Vec<f64>> {
        let mut vec_array = Vec::new();
        // for i in array.axis_iter(Axis(0)){
        //     vec_array.push(array.slice(s![.., i]).)
        // }
        for (i, mut row) in array.axis_iter(Axis(0)).enumerate() {
            let mut column_vector = Vec::new();
            for (j, col) in row.iter().enumerate() {
                column_vector.push(array[[i,j]]);
            }
            vec_array.push(column_vector.to_vec());
            column_vector.clear();
        }
        return vec_array
    }

    // NUMPY WAY
    // for i in (0..(frames.size() - increment)).step_by(frame_stack_stride) {
    //     stop_idx = i + increment;
    //     for start_time in (i..stop_idx).step_by(frame_stack_stride) {
    //         let mut sth = frames.row(start_time as Ix);
    //         stack = stack(Axis(0), &[stack.view(), sth.view()])
    //     }
    //     stacked_frames = stack(Axis(1), &[stacked_frames.view(), stack.view()])
    // }
    // return frames


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
            frame_stack_stride: 9,
            noti: 12,
        };
        let x = array!([0.,1.],[2.,3.]);
        let y = array!([4.,5.],[6.,7.]);
        let outcome = test_object.compute_internal(x.view().into_dyn(),y.view().into_dyn());
        assert_eq!(array!([6.0]), outcome)
    }

}