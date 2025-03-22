mod forest;
mod node;

use forest::IsolationForest;

fn main() {
    let data = vec![
        vec![1.0, 2.0],
        vec![1.1, 2.2],
        vec![1.2, 2.1],
        vec![1.3, 2.0],
        vec![1.2, 2.3],
        vec![10.0, 20.0], // outlier!
    ];

    // Create a forest with 100 trees, subsample size of 4, and a random seed
    let mut forest = IsolationForest::new(100, 4, Some(42));

    // Fit the forest
    forest.fit(&data);

    // Score
    let scores = forest.score(&data);

    // Report
    println!("Data points and their anomaly scores:");
    for (i, (point, score)) in data.iter().zip(scores.iter()).enumerate() {
        println!("Point {}: {:?} - Score: {:.6}", i, point, score);
    }

    println!("\nHigher scores (closer to 1.0) indicate anomalies");
}
