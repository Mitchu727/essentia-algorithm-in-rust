use rulinalg::utils::{argmax, dot};
use crate::{normalize, sum_frames};

pub fn chroma_cross_binary_sim_matrix(chroma_a: Vec<Vec<f64>>, chroma_b: Vec<Vec<f64>>, n_shifts: u32, match_coefficient:f64, mismatch_coefficient
: f64) -> Vec<Vec<f64>> {
    let mut value_at_shifts = Vec::new();
    let mut oti_index;
    let mut sim_matrix = generate_two_dimensional_array(chroma_a.len(), chroma_b.len());
    for i in 0..chroma_a.len() {
        for j in 0..chroma_b.len() {
            for k in 0..n_shifts {
                let mut chroma_b_copy = chroma_b[j].clone();
                chroma_b_copy.rotate_right(k as usize);
                value_at_shifts.push(dot(&chroma_b_copy, &chroma_a[i]));
            }
            oti_index = argmax(&value_at_shifts);
            value_at_shifts.clear();
            if oti_index.0 == 0 || oti_index.0 == 1 {
                sim_matrix[i][j] = match_coefficient
            }
            else {
                sim_matrix[i][j] = mismatch_coefficient
            }
        }
    }
    return sim_matrix
}

pub fn stack_chroma_frames(frames: Vec<Vec<f64>>, frame_stack_size: usize, frame_stack_stride: usize) -> Vec<Vec<f64>> {
    if frame_stack_size == 1 {
        return frames;
    }

    let increment = frame_stack_size + frame_stack_stride;
    let mut stacked_frames = Vec::new();
    let mut stack = Vec::new();
    let mut stop_idx: usize;

    for i in (0..(frames.len() - increment)).step_by(frame_stack_stride) {
        stop_idx = i + increment;
        for start_time in (i..stop_idx).step_by(frame_stack_stride) {
            stack.extend_from_slice(&frames[start_time])
        }
        stacked_frames.push(stack.clone());
        stack.clear()
    }
    return stacked_frames;
}

pub fn get_columns_values_at_vec_index (input_matrix: Vec<Vec<f64>>, index: usize) -> Vec<f64> {
    let mut row = Vec::new();
    for i in 0..input_matrix.len() {
        row.push(input_matrix[i][index])
    }
    return row
}

pub fn optimal_transposition_index(chroma_a: Vec<Vec<f64>>, chroma_b: Vec<Vec<f64>>, n_shifts: u32) -> usize {
    let global_chroma_a= global_average_chroma(chroma_a);
    let mut global_chroma_b = global_average_chroma(chroma_b);
    let mut value_at_shifts = Vec::new();
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

pub fn generate_two_dimensional_array(width: usize, height: usize) -> Vec<Vec<f64>> {
    let mut array = Vec::new();
    for _i in 0..width {
        array.push(vec![0.; height]);
    }
    return array
}