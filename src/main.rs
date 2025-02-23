use burn::tensor::{backend::NdArray, Data, Tensor}; 
use burn::module::Module;
use burn::optim::{Optimizer, SGD}; 
use burn::nn::loss::mse_loss;
use rand::Rng;
use textplots::{Chart, Plot, Shape};


#[derive(Module, Debug)]
struct LinearRegression {
    weight: Tensor<NdArray, 1>, 
    bias: Tensor<NdArray, 1>,
}

impl LinearRegression {
    fn new() -> Self {
        Self {
            weight: Tensor::from_floats([0.0]),  // Start with random weights
            bias: Tensor::from_floats([0.0]), 
        }
    }

    fn forward(&self, x: &Tensor<NdArray, 1>) -> Tensor<NdArray, 1> {
        self.weight.clone() * x.clone() + self.bias.clone()
    }
}

fn generate_data() -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_vals = vec![];
    let mut y_vals = vec![];

    for _ in 0..100 {
        let x: f32 = rng.gen_range(0.0..10.0);
        let y: f32 = 2.0 * x + 1.0 + rng.gen_range(-1.0..1.0);
        x_vals.push(x);
        y_vals.push(y);
    }

    (x_vals, y_vals)
}

fn main() {
    let (x_vals, y_vals) = generate_data();

    let x_tensor = Tensor::<NdArray, 1>::from_data(Data::from(x_vals.clone()));
    let y_tensor = Tensor::<NdArray, 1>::from_data(Data::from(y_vals.clone()));

    let mut model = LinearRegression::new();
    let mut optimizer = SGD::new(0.01);  // Learning rate

    for epoch in 0..100 {  // Training loop
        let predictions = model.forward(&x_tensor);
        let loss = mse_loss(&predictions, &y_tensor);

        optimizer.step(loss.backward());

        if epoch % 10 == 0 {  // Print every 10 epochs
            println!("Epoch {}: Loss = {:?}", epoch, loss.into_scalar());
        }
    }

    println!("Training complete!");
}

fn test_model(model: &LinearRegression) {
    let test_x: Vec<f32> = (0..10).map(|x| x as f32).collect();
    let test_y: Vec<f32> = test_x.iter().map(|x| 2.0 * x + 1.0).collect();
    let pred_y: Vec<f32> = model
        .forward(&Tensor::<NdArray, 1>::from_data(Data::from(test_x.clone())))
        .into_data()
        .convert();

    println!("\nTesting Model Predictions:");
    for (x, pred) in test_x.iter().zip(pred_y.iter()) {
        println!("For x = {:.1}, predicted y = {:.2}", x, pred);
    }

    println!("\nPlotting Results...");
    Chart::new(100, 30, 0.0, 10.0)
        .lineplot(&Shape::Lines(&test_x, &test_y))
        .lineplot(&Shape::Lines(&test_x, &pred_y))
        .display();
}


fn main() {
    // ... (Previous training code)

    test_model(&model);
}


