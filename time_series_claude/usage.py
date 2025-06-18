# Create and train a model
forecaster = TimeSeriesForecaster(
    model_type='lstm',  # or 'gru', 'transformer'
    seq_length=60,      # lookback window
    pred_length=1,      # forecast horizon
    hidden_size=64,
    num_layers=2
)

# Prepare your data (numpy array)
train_dataset, test_dataset = forecaster.prepare_data(your_data)

# Train
forecaster.train(train_dataset, epochs=100)

# Evaluate
metrics = forecaster.evaluate(test_dataset)