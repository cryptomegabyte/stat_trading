#[derive(Debug, Clone)]
pub struct TradeData {
    pub price: f64,
    pub volume: f64,
}

#[derive(Debug)]
pub enum MLModel {
    LinearRegression(linfa_linear::FittedLinearRegression<f64>),
    RandomForest(smartcore::ensemble::random_forest_regressor::RandomForestRegressor<f64, f64, smartcore::linalg::basic::matrix::DenseMatrix<f64>, Vec<f64>>),
}