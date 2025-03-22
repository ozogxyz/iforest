use rand::rngs::StdRng;
use rand::{SeedableRng, Rng};
use std::cmp::min;

use crate::node::IsolationTreeNode;

pub struct IsolationForest {
    trees: Vec<IsolationTreeNode>,
    num_trees: usize,
    subsample_size: usize,
    max_tree_height: usize,
    rng: StdRng,
}

impl IsolationForest {
    pub fn new(num_trees: usize, subsample_size: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        let max_tree_height = (subsample_size as f64).log2().ceil() as usize;

        IsolationForest {
            trees: Vec::with_capacity(num_trees),
            num_trees,
            subsample_size,
            max_tree_height,
            rng,
        }
    }

    pub fn fit(&mut self, data: &[Vec<f64>]) {
        self.trees.clear();

        for _ in 0..self.num_trees {
            let subsample = self.get_random_subsample(data);

            // Build a tree with the subsample
            let tree = IsolationTreeNode::build_isolation_tree(
                &subsample,
                0,
                self.max_tree_height,
                &mut self.rng,
            );

            // Add the tree to the forest
            self.trees.push(tree);
        }
    }

    // Take a random subsample of the data
    fn get_random_subsample(&mut self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let data_size = data.len();
        let sample_size = min(self.subsample_size, data_size);

        // If sample size == data size just clone the whole dataset
        if sample_size == data_size {
            return data.to_vec();
        }

        // Create a set of randomly selected indices
        let mut indices = Vec::with_capacity(sample_size);
        while indices.len() < sample_size {
            let idx = self.rng.random_range(0..data_size);
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }

        // Create subsample from selected indices
        indices.iter().map(|&idx| data[idx].clone()).collect()
    }

    // Scoring function - return scoring for each data point
    pub fn score(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter()
            .map(|instance| self.score_instance(instance))
            .collect()
    }

    // Scores a single data point
    pub fn score_instance(&self, instance: &[f64]) -> f64 {
        let avg_path_len = self.avg_path_len(instance);

        // Normalize by expected path length in successful search
        let norm_factor = IsolationTreeNode::c(self.subsample_size);

        // Convert to anomaly score (2^(-avg_path_length/normalizing_factor))
        // Higher scores (closer to 1) indicate anomalies
        2.0f64.powf(-avg_path_len / norm_factor)
    }

    // Calculate average path length for an instance across all trees
    fn avg_path_len(&self, instance: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }

        // Sum path lengths for all trees
        let sum_path_len: f64 = self
            .trees
            .iter()
            .map(|tree| tree.path_len(instance, 0))
            .sum();

        sum_path_len / self.trees.len() as f64
    }
}
