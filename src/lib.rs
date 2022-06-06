mod essentia_math;

use numpy::ndarray::{Array, Array2, ArrayViewD, Axis, Ix2};
use numpy::{IntoPyArray, PyArray, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use pyo3::prelude::*;
use pyo3::create_exception;
use rulinalg::utils::{dot, argmax};
use essentia_math::{percentile, sum_frames, normalize, rotate_chroma, pairwise_distance};

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
}

#[pymethods]
impl ChromaCrossSimilarity {
    #[new]
    #[args(
    oti_binary = "false",
    frame_stack_size = "1",
    )]
    fn new(oti_binary: bool, frame_stack_size: usize) -> (Self, Algorithm) {
        (ChromaCrossSimilarity{oti_binary, frame_stack_size, frame_stack_stride: 1, noti: 12, oti: true, binarize_percentile: 0.095}, Algorithm::new())
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
        let mathc_coef = 1.;
        let mismatch_coef = 0.;
        if self.oti_binary {
            let stack_frames_a = stack_chroma_frames(query_feature, self.frame_stack_size, self.frame_stack_stride);
            let stack_frames_b = stack_chroma_frames(reference_feature, self.frame_stack_size, self.frame_stack_stride);
            return chroma_cross_binary_sim_matrix(stack_frames_a, stack_frames_b, self.noti, mathc_coef, mismatch_coef)
        }
        else {
            let query_feature_vecs = query_feature;
            let mut reference_feature_vecs = reference_feature;
            if self.oti {
                let oti_idx = optimal_transposition_index(query_feature_vecs.to_vec(), reference_feature_vecs.to_vec(), self.noti);
                rotate_chroma(&mut reference_feature_vecs, oti_idx as i32)
            }
            let query_feature_stack = stack_chroma_frames(query_feature_vecs.to_vec(), self.frame_stack_size, self.frame_stack_stride);
            let reference_feature_stack = stack_chroma_frames(reference_feature_vecs.to_vec(), self.frame_stack_size, self.frame_stack_stride);
            let p_distances = pairwise_distance(query_feature_stack, reference_feature_stack);
            let query_feature_size = p_distances.len();
            let reference_feature_size = p_distances[0].len();
            let mut threshold_reference = Vec::new();
            let mut threshold_query = Vec::new();
            // let mut csm = Array2::default([query_feature_size, reference_feature_size]);
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

fn stack_chroma_frames(frames: Vec<Vec<f64>>, frame_stack_size: usize, frame_stack_stride: usize) -> Vec<Vec<f64>> {
    if frame_stack_size == 1 {
        return frames;
    }
    let mut stop_idx: usize;
    let increment = frame_stack_size + frame_stack_stride;

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

fn get_columns_values_at_vec_index (input_matrix: Vec<Vec<f64>>, index: i32) -> Vec<f64> {
    // in essentia its called columns - depends on imagingation
    // TODO check if name conventions for columns and rows are good
    let mut row = Vec::new();
    for i in 0..input_matrix.len() {
        row.push(input_matrix[i][index as usize])
    }
    return row
}

fn optimal_transposition_index(chroma_a: Vec<Vec<f64>>, chroma_b: Vec<Vec<f64>>, n_shifts: u32) -> usize {
    let mut value_at_shifts = Vec::new();
    let global_chroma_a= global_average_chroma(chroma_a);
    let mut global_chroma_b = global_average_chroma(chroma_b);
    let mut iter_idx = 0;
    for i in 0..n_shifts {
        global_chroma_b.rotate_right((i - iter_idx) as usize);
        value_at_shifts.push(dot(&global_chroma_a, &global_chroma_b));
        if i >= 1 {
            iter_idx+=1;
        }
    }
    return argmax(&value_at_shifts).0
}

fn global_average_chroma(input_feature: Vec<Vec<f64>>) -> Vec<f64> {
    let mut global_chroma = sum_frames(input_feature);
    normalize(&mut global_chroma);
    return global_chroma;
}

fn chroma_cross_binary_sim_matrix(chroma_a: Vec<Vec<f64>>, chroma_b: Vec<Vec<f64>>, n_shifts: u32, match_coef:f64, mismatch_coef: f64) -> Vec<Vec<f64>> {
    let mut value_at_shifts = Vec::new();
    let mut oti_index;
    // let mut sim_matrix = Array2::default([chroma_a.len(), chroma_b.len()]);
    // let sim_matrix = [mut [mut 0f64, ..chroma_a.len()]]
    let mut sim_matrix = generate_two_dimensional_array(chroma_a.len(), chroma_b.len());
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
                 sim_matrix[i][j] = match_coef
            }
            else {
                sim_matrix[i][j] = mismatch_coef
            }
        }
    }
    return sim_matrix
}

fn generate_two_dimensional_array(height: usize, width: usize) -> (Vec<Vec<f64>>) {
    let mut array = Vec::new();
    let mut column = Vec::new();
    for i in 0..height {
        for j in 0..width {
            column.push(0.)
        }
        array.push(column.to_vec());
        column.clear();
    }
    return array
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
        // for i in array.axis_iter(Axis(0)){
        //     vec_array.push(array.slice(s![.., i]).)
        // }
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