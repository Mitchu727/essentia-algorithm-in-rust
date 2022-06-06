mod essentia_math;
mod chroma_cross_similarity_utils;

use numpy::ndarray::{Array2, ArrayViewD, Axis, Ix2};
use numpy::{IntoPyArray, PyArray, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use pyo3::prelude::*;
use pyo3::create_exception;
use essentia_math::{percentile, sum_frames, normalize, rotate_chroma, pairwise_distance};
use chroma_cross_similarity_utils::{stack_chroma_frames, };
use crate::chroma_cross_similarity_utils::{chroma_cross_binary_sim_matrix, generate_two_dimensional_array, get_columns_values_at_vec_index, optimal_transposition_index};

#[pyclass(subclass)]
struct Algorithm {
    #[pyo3(get)]
    processing_mode: String,
}

#[pymethods]
impl Algorithm {
    #[new]
    fn new() -> Self {
        Algorithm { processing_mode: String::from("Standard") }
    }
}

#[pyclass(extends=Algorithm, subclass)]
struct ChromaCrossSimilarity {
    #[pyo3(get)]
    oti_binary: bool,
    frame_stack_size: usize,
    frame_stack_stride: usize,
    noti: u32,
    oti: bool,
    binarize_percentile: f64,
    match_coefficient: f64,
    mismatch_coefficient: f64
}

#[pymethods]
impl ChromaCrossSimilarity {
    #[new]
    #[args(
    oti_binary = "false",
    frame_stack_size = "1",
    )]
    fn new(oti_binary: bool, frame_stack_size: usize) -> (Self, Algorithm) {
        (ChromaCrossSimilarity{
            oti_binary,
            frame_stack_size,
            frame_stack_stride: 1,
            noti: 12,
            oti: true,
            binarize_percentile: 0.095,
            match_coefficient:1.0,
            mismatch_coefficient: 0.0
        }, Algorithm::new())
    }

    fn __call__<'py>(&self, //na obecny moment może być statyczna, potem może się to zmienić
                    py: Python<'py>,
                    x: PyReadonlyArrayDyn<f64>,
                    y: PyReadonlyArrayDyn<f64>,
    ) -> Result<&'py PyArray<f64, Ix2>, PyErr> {
        self.py_compute(py,x,y)
    }

    #[pyo3(name = "compute")]
    fn py_compute<'py>(&self, //na obecny moment może być statyczna, potem może się to zmienić
                       py: Python<'py>,
                       x: PyReadonlyArrayDyn<f64>,
                       y: PyReadonlyArrayDyn<f64>,
    ) -> Result<&'py PyArray<f64, Ix2>, PyErr> {
        if x.ndim() != 2 || y.ndim() != 2 {
            return Err(EssentiaException::new_err("Wrong array dimensions"))
        }
        let x = x.as_array();
        let y = y.as_array();
        let z = from_vectors_to_ndarray(self.compute(from_ndarray_to_vectors(x),from_ndarray_to_vectors(y)));
        Ok(z.into_pyarray(py))
    }

}

impl ChromaCrossSimilarity{
    fn compute (&self, query_feature: Vec<Vec<f64>>, reference_feature: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        if self.oti_binary {
            let stack_frames_a = stack_chroma_frames(query_feature, self.frame_stack_size, self.frame_stack_stride);
            let stack_frames_b = stack_chroma_frames(reference_feature, self.frame_stack_size, self.frame_stack_stride);
            return chroma_cross_binary_sim_matrix(stack_frames_a, stack_frames_b, self.noti, self.match_coefficient, self.mismatch_coefficient)
        }
        else {
            if self.oti {
                let oti_idx = optimal_transposition_index(query_feature.to_vec(), reference_feature.to_vec(), self.noti);
                rotate_chroma(&mut reference_feature.to_vec(), oti_idx as i32)
            }
            let query_feature_stack = stack_chroma_frames(query_feature.to_vec(), self.frame_stack_size, self.frame_stack_stride);
            let reference_feature_stack = stack_chroma_frames(reference_feature.to_vec(), self.frame_stack_size, self.frame_stack_stride);
            let p_distances = pairwise_distance(query_feature_stack, reference_feature_stack);
            let query_feature_size = p_distances.len();
            let reference_feature_size = p_distances[0].len();
            let mut threshold_reference = Vec::new();
            let mut threshold_query = Vec::new();
            let mut csm = generate_two_dimensional_array(query_feature_size, reference_feature_size);
            for j in 0..reference_feature_size {
                let mut _status = true;
                for i in 0..query_feature_size {
                    if _status {
                        threshold_reference.push(percentile(get_columns_values_at_vec_index(p_distances.to_vec(), j as i32), 100.*self.binarize_percentile));
                    }
                    if p_distances[i][j] <= threshold_reference[j] {
                        csm[i][j] = 1.;
                    }
                    _status = false;
                }
            }
            for k in 0..query_feature_size {
                threshold_query.push(percentile(p_distances[k].to_vec(), 100.*self.binarize_percentile));
                for l in 0..reference_feature_size {
                    if p_distances[k][l] > threshold_query[k] {
                        csm[k][l] = 0.;
                    }
                }
            }
            return csm;
        }
    }
}

fn from_vectors_to_ndarray(vec_array :Vec<Vec<f64>>) -> Array2<f64> {
    let mut array = Array2::<f64>::default((vec_array.len(), vec_array[0].len()));
    for (i, mut row) in array.axis_iter_mut(Axis(0)).enumerate() {
        for (j, col) in row.iter_mut().enumerate() {
            *col = vec_array[i][j];
        }
    }
    return array
}

fn from_ndarray_to_vectors(array: ArrayViewD<f64>) -> Vec<Vec<f64>> {
        let mut vec_array = Vec::new();
        for (i, row) in array.axis_iter(Axis(0)).enumerate() {
            let mut column_vector = Vec::new();
            for (j, _col) in row.iter().enumerate() {
                column_vector.push(array[[i,j]]);
            }
            vec_array.push(column_vector.to_vec());
            column_vector.clear();
        }
        return vec_array
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

    // #[test]
    // fn simple_test() {
    //     let test_object = ChromaCrossSimilarity{
    //         oti_binary: false,
    //         frame_stack_size: 1,
    //         frame_stack_stride: 9,
    //         noti: 12,
    //         oti: false,
    //         binarize_percentile: 0.095
    //     };
    //     let x = array!([0.,1.],[2.,3.]);
    //     let y = array!([4.,5.],[6.,7.]);
    //     let outcome = test_object.compute_internal(x.view().into_dyn(),y.view().into_dyn());
    //     assert_eq!(array!([6.0]), outcome)
    // }

}