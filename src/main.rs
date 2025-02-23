use rand::Rng; //Random number generator

fn main() {
    let mut rng = rand::thread_rng();

    // Generate 100 (x, y) pairs with noise
    let data: Vec<(f32, f32)> = (0..100)
        .map(|_| {
            let x: f32 = rng.gen_range(0.0..10.0); // Random x value between 0 and 10
            let y: f32 = 2.0 * x + 1.0 + rng.gen_range(-1.0..1.0); // y = 2x + 1 with noise
            (x, y)
        })
        .collect();

    // Print some sample data points
    for (x, y) in data.iter().take(10) {
        println!("x: {:.2}, y: {:.2}", x, y);
    }
}

