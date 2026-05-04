# Capstone Project: Quantitative Options Trading with Machine Learning
This project applies machine learning to options trading, using two independently trained models to predict whether an SPY options contract will hit a +7% profit target before a -5% stop loss within a 21-calendar-day holding window. Historical data spans from 2018 through Q1 2026, totaling over 1.59 billion rows of minute-level options data. Both models were evaluated using walk-forward validation across multiple market years, with the strategy requiring a minimum win rate of 41.67% to break even.

## What is not included
Trained model artifacts and raw data files are not included in this repository. Models are derived from proprietary market data provided by ThetaData and cannot be redistributed. Raw data requires a valid ThetaData subscription to obtain.

## Want to learn more?
Full methodology, results, and findings can be found at [irvinoc.com](https://www.irvinoc.com/projects/quant-options-ml).