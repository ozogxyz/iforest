use rand::Rng;

#[derive(Debug)]
pub enum IsolationTreeNode {
    // Internal node that splits data
    Internal {
        split_feature: usize,
        split_value: f64,
        left: Box<IsolationTreeNode>,
        right: Box<IsolationTreeNode>,
        #[allow(dead_code)]
        depth: usize,
    },

    // Terminal node
    Terminal {
        size: usize,
        #[allow(dead_code)]
        depth: usize,
    },
}

impl IsolationTreeNode {
    pub fn path_len(&self, instance: &[f64], curr_len: usize) -> f64 {
        match self {
            // If @terminal node
            IsolationTreeNode::Terminal { size, .. } => {
                // Return the current path length + correction factor?
                curr_len as f64 + Self::c(*size)
            }
            // If @internal node
            IsolationTreeNode::Internal {
                split_feature,
                split_value,
                left,
                right,
                ..
            } => {
                // Decide which child to traverse based on split
                if instance[*split_feature] < *split_value {
                    left.path_len(instance, curr_len + 1)
                } else {
                    right.path_len(instance, curr_len + 1)
                }
            }
        }
    }

    // Helper function to calculate average path length correction factor
    // This is the expected path length in a Binary Search Tree
    pub fn c(size: usize) -> f64 {
        if size <= 1 {
            return 0.0;
        }

        let n = size as f64;
        // Euler's constant + harmonic number approximation
        2.0 * (n.ln() + 0.5772156649) - (2.0 * (n - 1.0) / n)
    }

    // Core of the algorithm
    pub fn build_isolation_tree(
        data: &[Vec<f64>],
        depth: usize,
        height_limit: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let sample_size = data.len();

        // Terminal conditions:
        // 1. Only one instance remains (perfectly isolated)
        // 2. All values in chosen attribute are identical (can't split further)
        // 3. Maximum height (depth) has been reached
        if sample_size <= 1 || depth >= height_limit {
            return IsolationTreeNode::Terminal {
                size: sample_size,
                depth,
            };
        }

        // Get number of features
        let num_features = if !data.is_empty() { data[0].len() } else { 0 };

        // Randomly select a feature to split on
        let split_feature = rng.random_range(0..num_features);

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for instance in data.iter() {
            let val = instance[split_feature];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // Check if all values are identical
        if (max_val - min_val).abs() < f64::EPSILON {
            return IsolationTreeNode::Terminal {
                size: sample_size,
                depth,
            };
        }

        // Generate a random split point between min and max
        let split_value = rng.random_range(min_val..=max_val);

        // Partition based on split
        let mut left_data = Vec::new();
        let mut right_data = Vec::new();

        for instance in data.iter() {
            if instance[split_feature] < split_value {
                left_data.push(instance.clone());
            } else {
                right_data.push(instance.clone());
            }
        }

        // If the split results in empty partitions, create a terminal node
        if left_data.is_empty() || right_data.is_empty() {
            return IsolationTreeNode::Terminal {
                size: sample_size,
                depth,
            };
        }

        // Recursively build left and right subtrees
        let left = Box::new(Self::build_isolation_tree(
            &left_data,
            depth + 1,
            height_limit,
            rng,
        ));

        let right = Box::new(Self::build_isolation_tree(
            &right_data,
            depth + 1,
            height_limit,
            rng,
        ));

        // Create and return an internal node
        IsolationTreeNode::Internal {
            split_feature,
            split_value,
            left,
            right,
            depth,
        }
    }
}
