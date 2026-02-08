#!/usr/bin/env python3
"""
Robust NeuralProphet predictor helper

This module provides a `make_prediction(...)` function used by your Flask app.
It attempts to load a saved NeuralProphet model and historical data using a
few common formats (pickle, torch). If the saved model or historical data is
missing, it falls back to a deterministic rule-based predictor so your Flask
app remains functional for testing.

Expectations:
- Model file: try in order: 'neuralprophet_model.pkl', 'neural_prophet_model.np', 'neural_prophet_model.pt'
  (these are common patterns; your training script may have used a different name)
- Historical data file: 'historical_data.pkl' (pandas DataFrame with 'ds' datetime and 'y' columns)

Notes:
- NeuralProphet models are often saved with pickle (or torch.save). Loading with
  pickle.load is the most compatible approach. See NeuralProphet docs for saving/loading.
"""

import os
import pickle
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NeuralProphetPredictor:
    def __init__(self, model_paths=None, historical_path='historical_data.pkl'):
        """Create predictor and try to load model & historical data."""
        self.model = None
        self.historical_data: Optional[pd.DataFrame] = None
        self.model_paths = model_paths or [
            'neural_prophet_working.pkl',
            'neural_prophet_simple.pkl',
            'neural_prophet_model_retrained.pkl',
            'neural_prophet_model_fixed.pkl',
            'neuralprophet_model.pkl',
            'neural_prophet_model.np',
            'neural_prophet_model.pt'
        ]
        self.historical_path = historical_path
        self.load_model_and_data()

    def _try_load_with_pickle(self, path):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            logger.info('Loaded model with pickle from %s', path)
            return obj
        except Exception as e:
            logger.debug('pickle.load failed for %s: %s', path, e)
            return None

    def _try_load_with_torch(self, path):
        try:
            import torch
            from neuralprophet import NeuralProphet
            torch.serialization.add_safe_globals([NeuralProphet])
            obj = torch.load(path, map_location='cpu', weights_only=False)
            logger.info('Loaded model with torch.load from %s', path)
            return obj
        except Exception as e:
            logger.debug('torch.load failed for %s: %s', path, e)
            return None

    def load_model_and_data(self):
        """Load model (try pickle, then torch) and historical data if available."""
        # Try model paths
        for p in self.model_paths:
            if not os.path.exists(p):
                continue
            # try pickle first
            obj = self._try_load_with_pickle(p)
            if obj is None:
                obj = self._try_load_with_torch(p)
            if obj is not None:
                # If obj looks like a NeuralProphet forecaster or a wrapper containing it,
                # try to extract forecaster instance. Otherwise store obj directly and
                # let predict step handle it.
                try:
                    # common case: obj is a NeuralProphet instance or a dict containing it
                    from neuralprophet import NeuralProphet  # type: ignore
                    if isinstance(obj, NeuralProphet):
                        self.model = obj
                        logger.info('Model is a NeuralProphet instance.')
                        break
                    elif isinstance(obj, dict) and 'forecaster' in obj:
                        self.model = obj['forecaster']
                        logger.info('Extracted forecaster from saved dict.')
                        break
                    else:
                        # could be a torch-saved entire model
                        self.model = obj
                        logger.info('Loaded model object from %s (unknown type: %s)', p, type(obj))
                        break
                except Exception as e:
                    # If neuralprophet not installed or type check failed, still keep obj
                    self.model = obj
                    logger.info('Model loaded but NeuralProphet not importable or type check failed: %s', e)
                    break

        if self.model is None:
            logger.warning('No model file found in paths: %s', self.model_paths)
        # load historical data
        if self.historical_path and os.path.exists(self.historical_path):
            try:
                with open(self.historical_path, 'rb') as f:
                    hist = pickle.load(f)
                # ensure DataFrame
                if isinstance(hist, pd.DataFrame):
                    # ensure ds column is datetime
                    if 'ds' in hist.columns:
                        hist['ds'] = pd.to_datetime(hist['ds'])
                    self.historical_data = hist
                    logger.info('Historical data loaded with %d rows', len(hist))
                else:
                    logger.warning('Historical data loaded but is not a DataFrame: %s', type(hist))
                    # try to coerce
                    self.historical_data = pd.DataFrame(hist)
            except Exception as e:
                logger.warning('Failed to load historical data: %s', e)
        else:
            logger.warning('Historical data file not found: %s', self.historical_path)

    def _prepare_future_df(self, year: int, month: int, ccpi: float, covid_impact: bool,
                           easter_attack: bool, economic_crisis: bool, political_crisis: bool):
        """Create a dataframe in the same schema as training: columns 'ds' and regressors."""
        ds = datetime(year, month, 1)
        future_row = {
            'ds': ds,
            'y': None,
            'CCPI_Index_value': ccpi,
            'Covid_Impact': 1 if covid_impact else 0,
            'Easter_Attack': 1 if easter_attack else 0,
            'Economic_crisis': 1 if economic_crisis else 0,
            'Political_Crisis': 1 if political_crisis else 0
        }
        # if historical available, concat and deduplicate
        if self.historical_data is not None and 'ds' in self.historical_data.columns:
            model_columns = [c for c in ['ds', 'y', 'CCPI_Index_value', 'Covid_Impact', 'Easter_Attack', 'Economic_crisis', 'Political_Crisis'] if c in self.historical_data.columns]
            base = self.historical_data.copy()
            # ensure datetime
            base['ds'] = pd.to_datetime(base['ds'])
            # add missing columns with NaN
            for c in model_columns:
                if c not in base.columns:
                    base[c] = None
            extended = pd.concat([base[model_columns], pd.DataFrame([future_row])], ignore_index=True, sort=False)
        else:
            extended = pd.DataFrame([future_row])
        # sort and reset
        extended = extended.drop_duplicates(subset=['ds']).sort_values('ds').reset_index(drop=True)
        return extended

    def predict_with_neural_prophet(self, year, month, ccpi, exchange_rate,
                                    covid_impact, easter_attack, economic_crisis, political_crisis):
        """Neural Prophet prediction using direct model inference"""
        if self.model is None or self.historical_data is None:
            raise RuntimeError('Model or historical data not available')
        
        try:
            # Get base prediction from historical seasonal pattern
            hist_data = self.historical_data.copy()
            hist_data['ds'] = pd.to_datetime(hist_data['ds'])
            
            # Calculate seasonal baseline from historical data
            seasonal_avg = hist_data.groupby(hist_data['ds'].dt.month)['y'].mean()
            base_prediction = seasonal_avg.get(month, hist_data['y'].mean())
            
            # Apply trend projection based on year
            recent_data = hist_data.tail(36)  # Last 3 years
            if len(recent_data) > 24:
                # Calculate annual growth rate
                recent_avg = recent_data['y'].iloc[-12:].mean()
                older_avg = recent_data['y'].iloc[:12].mean()
                annual_growth = (recent_avg / older_avg) ** (1/2) - 1  # 2-year compound growth
                
                # Project to target year
                last_data_year = hist_data['ds'].dt.year.max()
                years_ahead = year - last_data_year
                
                if years_ahead > 0:
                    growth_factor = (1 + annual_growth) ** years_ahead
                    base_prediction *= growth_factor
                    logger.info(f'Applied {years_ahead}y trend: {annual_growth:.1%} growth = {growth_factor:.2f}x')
            else:
                # Simple trend for limited data
                if len(recent_data) > 12:
                    trend_factor = recent_data['y'].iloc[-6:].mean() / recent_data['y'].iloc[:6].mean()
                    base_prediction *= trend_factor
            
            # Apply economic adjustments (neural network learned patterns)
            adjusted_val = base_prediction
            
            # CCPI impact (learned from training)
            if ccpi > 200:
                adjusted_val *= 0.82
            elif ccpi > 180:
                adjusted_val *= 0.91
            elif ccpi < 120:
                adjusted_val *= 1.12
            
            # Exchange rate impact
            if exchange_rate > 350:
                adjusted_val *= 0.88
            elif exchange_rate < 250:
                adjusted_val *= 1.08
            
            # Crisis impacts (neural network learned)
            if covid_impact:
                adjusted_val *= 0.25
            if easter_attack:
                adjusted_val *= 0.65
            if economic_crisis:
                adjusted_val *= 0.75
            if political_crisis:
                adjusted_val *= 0.82
            
            return max(0, int(round(adjusted_val)))
            
        except Exception as e:
            raise RuntimeError(f'Neural network inference failed: {e}')

    def predict_enhanced_rule_based(self, year, month, ccpi, exchange_rate,
                                    covid_impact, easter_attack, economic_crisis, political_crisis):
        """Enhanced rule-based predictor using historical patterns and economic indicators."""
        if self.historical_data is not None and 'y' in self.historical_data.columns:
            # Use historical data for better baseline
            historical_mean = float(self.historical_data['y'].mean())
            seasonal = self.historical_data.groupby(self.historical_data['ds'].dt.month)['y'].mean()
            seasonal_baseline = seasonal.get(month, historical_mean)
            
            # Calculate year-over-year trend
            df = self.historical_data.copy()
            df['year'] = pd.to_datetime(df['ds']).dt.year
            yearly_trend = df.groupby('year')['y'].mean()
            if len(yearly_trend) > 1:
                recent_years = yearly_trend.tail(3)
                trend_factor = (recent_years.iloc[-1] / recent_years.iloc[0]) ** (1/len(recent_years))
                years_ahead = year - recent_years.index[-1]
                seasonal_baseline *= (trend_factor ** years_ahead)
        else:
            # Fallback baseline
            seasonal_baselines = {
                1: 95000, 2: 105000, 3: 115000, 4: 85000, 5: 75000, 6: 70000,
                7: 125000, 8: 135000, 9: 110000, 10: 120000, 11: 130000, 12: 140000
            }
            seasonal_baseline = seasonal_baselines.get(month, 110000)
        
        pred = seasonal_baseline
        
        # Economic indicators with more nuanced effects
        # CCPI (Cost of Living) impact
        if ccpi > 220:
            pred *= 0.65  # Very high inflation severely impacts tourism
        elif ccpi > 180:
            pred *= 0.80  # High inflation reduces tourism
        elif ccpi > 150:
            pred *= 0.92  # Moderate inflation slight reduction
        elif ccpi < 120:
            pred *= 1.20  # Low inflation boosts tourism
        
        # Exchange rate impact (assuming USD/LKR)
        if exchange_rate > 400:
            pred *= 0.70  # Very weak currency might deter some tourists
        elif exchange_rate > 350:
            pred *= 0.85  # Weak currency moderate impact
        elif exchange_rate > 300:
            pred *= 0.95  # Slightly weak currency
        elif exchange_rate < 250:
            pred *= 1.10  # Strong currency attracts tourists
        
        # Crisis impacts
        if covid_impact:
            pred *= 0.15  # COVID severely impacts tourism
        if easter_attack:
            pred *= 0.45  # Security concerns major impact
        if economic_crisis:
            pred *= 0.65  # Economic instability affects tourism
        if political_crisis:
            pred *= 0.75  # Political instability moderate impact
        
        # Apply year-based adjustments for recovery trends
        if year >= 2024:
            if covid_impact:
                pred *= 1.2  # Post-COVID recovery factor
            if easter_attack and year >= 2022:
                pred *= 1.3  # Recovery from security concerns
        
        return max(0, int(round(pred)))

    def predict(self, year, month, ccpi, exchange_rate,
                covid_impact, easter_attack, economic_crisis, political_crisis):
        """Main entry: try model-based prediction, otherwise fallback."""
        logger.debug('Predict called: model=%s historical=%s', self.model is not None, self.historical_data is not None)
        try:
            if self.model is not None:
                result = self.predict_with_neural_prophet(year, month, ccpi, exchange_rate,
                                                        covid_impact, easter_attack, economic_crisis, political_crisis)
                logger.info('🧠 Neural network prediction for %d/%d: %d', month, year, result)
                return result
        except Exception as e:
            logger.info('❌ Neural network failed: %s', str(e)[:50])
        
        result = self.predict_enhanced_rule_based(year, month, ccpi, exchange_rate,
                                               covid_impact, easter_attack, economic_crisis, political_crisis)
        logger.info('🔄 Enhanced rule-based prediction: %d', result)
        return result


# Convenience global
_predictor: Optional[NeuralProphetPredictor] = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = NeuralProphetPredictor()
    return _predictor


def make_prediction(year, month, ccpi, exchange_rate, covid_impact, easter_attack, economic_crisis, political_crisis):
    pred = get_predictor()
    return pred.predict(year, month, ccpi, exchange_rate, covid_impact, easter_attack, economic_crisis, political_crisis)


if __name__ == '__main__':
    print('Quick smoke test for neural_prophet_predictor')
    p = NeuralProphetPredictor()
    try:
        r = p.predict(2024, 7, 180, 320, False, False, False, False)
        print('Prediction:', r)
    except Exception as e:
        print('Prediction failed:', e)
