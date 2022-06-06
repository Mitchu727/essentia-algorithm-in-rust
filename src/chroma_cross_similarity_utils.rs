use rulinalg::utils::{argmax, dot};
use crate::{normalize, sum_frames};

pub fn chroma_cross_binary_sim_matrix(chroma_a: Vec<Vec<f64>>, chroma_b: Vec<Vec<f64>>, n_shifts: u32, match_coef:f64, mismatch_coef: f64) -> Vec<Vec<f64>> {
    let mut value_at_shifts = Vec::new();
    let mut oti_index;
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

pub fn stack_chroma_frames(frames: Vec<Vec<f64>>, frame_stack_size: usize, frame_stack_stride: usize) -> Vec<Vec<f64>> {
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

pub fn get_columns_values_at_vec_index (input_matrix: Vec<Vec<f64>>, index: i32) -> Vec<f64> {
    // in essentia its called columns - depends on imagingation
    // TODO check if name conventions for columns and rows are good
    let mut row = Vec::new();
    for i in 0..input_matrix.len() {
        row.push(input_matrix[i][index as usize])
    }
    return row
}

pub fn optimal_transposition_index(chroma_a: Vec<Vec<f64>>, chroma_b: Vec<Vec<f64>>, n_shifts: u32) -> usize {
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

pub fn global_average_chroma(input_feature: Vec<Vec<f64>>) -> Vec<f64> {
    let mut global_chroma = sum_frames(input_feature);
    normalize(&mut global_chroma);
    return global_chroma;
}

pub fn generate_two_dimensional_array(height: usize, width: usize) -> Vec<Vec<f64>> {
    let mut array = Vec::new();
    let mut column = Vec::new();
    for _i in 0..height {
        for _j in 0..width {
            column.push(0.)
        }
        array.push(column.to_vec());
        column.clear();
    }
    return array
}