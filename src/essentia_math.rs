use rulinalg::utils::{dot};

pub fn percentile(array: &Vec<f64>, mut q_percentile: f64) -> f64 {
    let mut sorted_array = array.clone();
    sorted_array.sort_by(|a, b| a.partial_cmp(b).unwrap());
    q_percentile /= 100.;
    let k;
    if sorted_array.len() > 1 {
        k = (sorted_array.len() - 1) as f64 * q_percentile;
    } else {
        k = sorted_array.len() as f64 * q_percentile;
    }
    let d0 = sorted_array[k.floor() as usize] * (k.ceil() - k);
    let d1 = sorted_array[k.ceil() as usize] * (k - k.floor());
    return d0 + d1
}

pub fn pairwise_distance(m: Vec<Vec<f64>>,n: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut pairwise_distance = Vec::new();
    let mut pairwise_distance_column = Vec::new();
    for i in 0..m.len() {
        for j in 0..n.len() {
            let item = dot(&m[i], &m[i]) + dot(&n[j], &n[j]) - 2.*dot(&m[i], &n[j]);
            pairwise_distance_column.push(item.sqrt())
        }
        pairwise_distance.push(pairwise_distance_column.clone());
        pairwise_distance_column.clear();
    }
    return pairwise_distance;
}

pub fn rotate_chroma(input_matrix: &mut Vec<Vec<f64>>, oti: usize) {
    for i in 0..input_matrix.len() {
        input_matrix[i].rotate_right(oti)
    }
}

pub fn normalize(array: &mut Vec<f64>) {
    if array.is_empty() {
        return
    }
    let max_element = array.iter().copied().fold(f64::NAN, f64::max);
    if max_element != 0. {
        for i in 0..array.len() {
            array[i] /= max_element;
        }
    }
}

pub fn sum_frames(frames: &Vec<Vec<f64>>) -> Vec<f64> {
    let number_of_frames = frames.len();
    let vector_size = frames[0].len();
    let mut result = vec![0.; vector_size];
    for j in 0..vector_size {
        for i in 0..number_of_frames {
            result[j] += frames[i][j]
        }
    }
    return result
}
