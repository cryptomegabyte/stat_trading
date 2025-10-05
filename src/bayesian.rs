use std::collections::HashMap;
use statrs::distribution::{Beta, Normal, ContinuousCDF};
use statrs::function::beta::beta;

/// Bayesian analysis utilities for trading model comparison and uncertainty quantification
#[derive(Debug)]
pub struct BayesianAnalyzer {
    /// Prior beliefs about model performance (beta distribution parameters)
    model_priors: HashMap<String, (f64, f64)>, // (alpha, beta) for beta distribution
}

impl Default for BayesianAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianAnalyzer {
    pub fn new() -> Self {
        let mut model_priors = HashMap::new();

        // Default priors: Beta(2, 2) - slightly optimistic about model performance
        // This represents prior belief of 2 wins and 2 losses
        model_priors.insert("random_forest".to_string(), (2.0, 2.0));
        model_priors.insert("gas_ghd".to_string(), (2.0, 2.0));
        model_priors.insert("gas_vg".to_string(), (2.0, 2.0));
        model_priors.insert("gas_nig".to_string(), (2.0, 2.0));
        model_priors.insert("gas_gld".to_string(), (2.0, 2.0));
        model_priors.insert("hybrid_egarch_lstm".to_string(), (2.0, 2.0));

        Self { model_priors }
    }

    /// Calculate Bayes factor comparing two models
    /// BF > 1 favors model1, BF < 1 favors model2
    pub fn bayes_factor(&self, model1: &str, model2: &str, data1: &ModelPerformance, data2: &ModelPerformance) -> f64 {
        let prior1 = self.model_priors.get(model1).unwrap_or(&(2.0, 2.0));
        let prior2 = self.model_priors.get(model2).unwrap_or(&(2.0, 2.0));

        let marg_like1 = self.beta_binomial_marginal_likelihood(data1.wins, data1.total_trades, prior1.0, prior1.1);
        let marg_like2 = self.beta_binomial_marginal_likelihood(data2.wins, data2.total_trades, prior2.0, prior2.1);

        // Assuming equal prior model probabilities for now
        marg_like1 / marg_like2
    }

    /// Compute marginal likelihood for beta-binomial model
    fn beta_binomial_marginal_likelihood(&self, wins: usize, total: usize, alpha: f64, beta_param: f64) -> f64 {
        let losses = total - wins;

        // Marginal likelihood = ∫ p(data|θ) p(θ) dθ
        // For beta-binomial: B(alpha+wins, beta+losses) / B(alpha, beta)
        let beta_func_prior = beta(alpha, beta_param);
        let beta_func_post = beta(alpha + wins as f64, beta_param + losses as f64);

        beta_func_post / beta_func_prior
    }

    /// Calculate posterior probability that model1 is better than model2
    pub fn posterior_model_probability(&self, model1: &str, model2: &str, data1: &ModelPerformance, data2: &ModelPerformance) -> f64 {
        let bf = self.bayes_factor(model1, model2, data1, data2);
        bf / (bf + 1.0)
    }

    /// Calculate credible interval for win rate
    pub fn win_rate_credible_interval(&self, model: &str, data: &ModelPerformance, credibility: f64) -> (f64, f64) {
        let prior = self.model_priors.get(model).unwrap_or(&(2.0, 2.0));

        let post_alpha = prior.0 + data.wins as f64;
        let post_beta = prior.1 + (data.total_trades - data.wins) as f64;

        let beta_dist = Beta::new(post_alpha, post_beta).unwrap();

        let lower_quantile = (1.0 - credibility) / 2.0;
        let upper_quantile = 1.0 - lower_quantile;

        (
            beta_dist.inverse_cdf(lower_quantile),
            beta_dist.inverse_cdf(upper_quantile)
        )
    }

    /// Calculate expected P&L with uncertainty
    pub fn pnl_credible_interval(&self, data: &ModelPerformance, credibility: f64) -> (f64, f64) {
        if data.total_trades == 0 {
            return (0.0, 0.0);
        }

        let mean_pnl = data.total_pnl / data.total_trades as f64;
        let variance = data.pnl_variance.unwrap_or(100.0);

        let std_dev = variance.sqrt() / (data.total_trades as f64).sqrt();

        let normal_dist = Normal::new(mean_pnl, std_dev).unwrap();

        let lower_quantile = (1.0 - credibility) / 2.0;
        let upper_quantile = 1.0 - lower_quantile;

        (
            normal_dist.inverse_cdf(lower_quantile),
            normal_dist.inverse_cdf(upper_quantile)
        )
    }

    /// Bayesian model averaging weights for ensemble predictions
    pub fn bma_weights(&self, models: &[(&str, &ModelPerformance)]) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        let mut total_weight = 0.0;

        for (model_name, performance) in models {
            let prior = self.model_priors.get(*model_name).unwrap_or(&(2.0, 2.0));
            let marg_like = self.beta_binomial_marginal_likelihood(
                performance.wins,
                performance.total_trades,
                prior.0,
                prior.1
            );
            weights.insert(model_name.to_string(), marg_like);
            total_weight += marg_like;
        }

        // Normalize to get posterior model probabilities
        for weight in weights.values_mut() {
            *weight /= total_weight;
        }

        weights
    }
}

/// Performance data for a trading model
#[derive(Debug, Clone)]
pub struct ModelPerformance {
    pub wins: usize,
    pub total_trades: usize,
    pub total_pnl: f64,
    pub pnl_variance: Option<f64>,
}

impl ModelPerformance {
    pub fn new(wins: usize, total_trades: usize, total_pnl: f64) -> Self {
        Self {
            wins,
            total_trades,
            total_pnl,
            pnl_variance: None,
        }
    }

    pub fn with_variance(mut self, variance: f64) -> Self {
        self.pnl_variance = Some(variance);
        self
    }

    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.wins as f64 / self.total_trades as f64
        }
    }

    pub fn average_pnl(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.total_pnl / self.total_trades as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayes_factor_calculation() {
        let analyzer = BayesianAnalyzer::new();

        let perf1 = ModelPerformance::new(60, 100, 1200.0);
        let perf2 = ModelPerformance::new(50, 100, 1000.0);

        let bf = analyzer.bayes_factor("gas_gld", "random_forest", &perf1, &perf2);
        assert!(bf > 1.0);
    }

    #[test]
    fn test_credible_interval() {
        let analyzer = BayesianAnalyzer::new();
        let perf = ModelPerformance::new(60, 100, 1200.0);

        let (lower, upper) = analyzer.win_rate_credible_interval("gas_gld", &perf, 0.95);

        assert!(lower > 0.0 && lower < upper && upper < 1.0);
        assert!(lower <= 0.6 && upper >= 0.6);
    }
}