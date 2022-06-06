use rulinalg::utils::{dot};

pub fn percentile(array: Vec<f64>, mut q_percentile: f64) -> f64 {
    //  if (array.empty())
    //    throw EssentiaException("percentile: trying to calculate percentile of empty array");
    let mut sorted_array = array;
    sorted_array.sort_by(|a, b| a.partial_cmp(b).unwrap());
    print!("{:?}/n", sorted_array);
    q_percentile /= 100.;
    let sorted_array_size = sorted_array.len();
    let k;
    if sorted_array_size > 1 {
        k = (sorted_array_size - 1) as f64 * q_percentile;
    } else {
        k = sorted_array_size as f64 * q_percentile;
    }
    print!("{}\n", k);
    let d0 = sorted_array[k.floor() as usize] * (k.ceil() - k);
    let d1 = sorted_array[k.ceil() as usize] * (k - k.floor());
    return d0 + d1
}

pub fn pairwise_distance(m: Vec<Vec<f64>>,n: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    // if m.is_empty() || n.is_empty() {
    //     throw EssentiaException("pairwiseDistance: found empty array as input!");
    // }

    let mut pdist = Vec::new();
    let mut pdist_column = Vec::new();
    for i in 0..m.len() {
        for j in 0..n.len() {
            let item = dot(&m[i], &m[i]) + dot(&n[j], &n[j]) - 2.*dot(&m[i], &n[j]);
            pdist_column.push(item.sqrt())
        }
        pdist.push(pdist_column.to_vec());
        pdist_column.clear();
    }
    // if (pdist.empty())
    // throw EssentiaException("pairwiseDistance: outputs an empty similarity matrix!");
    return pdist;
}

pub fn rotate_chroma(input_matrix: &mut Vec<Vec<f64>>, oti: i32) {
    // if (inputMatrix.empty())
    // throw EssentiaException("rotateChroma: trying to rotate an empty matrix");
    for i in 0..input_matrix.len() {
        input_matrix[i].rotate_right(oti as usize)
    }
}

pub fn normalize(array: &mut Vec<f64>) {
    if array.is_empty() {
        return
    }
    // let max_element = array.iter().max_by(|a, b| a.total_cmp(b));
    let max_element = array.iter().copied().fold(f64::NAN, f64::max);
    if max_element != 0. {
        for i in 0..array.len() {
            array[i] /= max_element;
        }
    }
}

pub fn sum_frames(frames: Vec<Vec<f64>>) -> Vec<f64> {
    // if (frames.empty()) {
    //     throw EssentiaException("sumFrames: trying to calculate sum of empty input frames");
    // }
    let number_of_frames = frames.len();
    let vsize = frames[0].len();
    let mut result = vec![0.; vsize];
    for j in 0..vsize {
        for i in 0..number_of_frames {
            result[j] += frames[i][j]
        }
    }
    return result
}