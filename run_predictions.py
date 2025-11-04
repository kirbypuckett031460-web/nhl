from nhl_model import NHLOverUnderModel, create_sample_data

# Example usage
if __name__ == "__main__":
    print("NHL Over/Under Betting Model")
    print("="*40)
    
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data()
    
    # Initialize and train model
    model = NHLOverUnderModel()
    
    print("Creating features...")
    df_features = model.create_features(df)
    
    print("Preparing model data...")
    X, y = model.prepare_model_data(df_features)
    
    print("Training ensemble model...")
    X_test, y_test, predictions = model.train_model(X, y, model_type='ensemble')
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    importance = model.get_feature_importance()
    print(importance.head(10))
    
    # Example prediction with enhanced features
    print("\nExample Game Prediction with Goaltender & Venue Analysis:")
    sample_features = X.iloc[0].values
    predicted_total = model.predict_game(sample_features)
    betting_line = 6.5
    
    recommendation, edge, explanation = model.betting_recommendation(predicted_total, betting_line)
    
    print(f"Predicted Total Goals: {predicted_total:.2f}")
    print(f"Betting Line: {betting_line}")
    print(f"Recommendation: {recommendation}")
    print(f"Edge: {edge:+.2f} goals")
    print(f"Explanation: {explanation}")
    
    # Show model insights
    print(f"\nModel Analysis:")
    print(f"- Goaltender factors included: Save % and GAA trends")
    print(f"- Venue factors: Altitude, dome effects, travel distance")
    print(f"- Enhanced features: {len(model.feature_names)} total variables")
    print(f"- Model accounts for goaltender fatigue and matchup quality")
    
    # Additional example: Multiple game predictions
    print(f"\n" + "="*50)
    print("TONIGHT'S SAMPLE PREDICTIONS")
    print("="*50)
    
    # Simulate tonight's games
    sample_games = [
        ("TOR @ BOS", 6.5),
        ("EDM @ COL", 7.0),
        ("NYR @ FLA", 6.0)
    ]
    
    for game, line in sample_games:
        # Use a random sample from our test data
        sample_idx = len(predictions) // 3  # Use different samples
        sample_features = X_test.iloc[sample_idx].values
        predicted = model.predict_game(sample_features)
        rec, edge, exp = model.betting_recommendation(predicted, line)
        
        print(f"\n{game}")
        print(f"Line: {line} | Prediction: {predicted:.2f}")
        print(f"Recommendation: {rec} ({edge:+.2f})")
        
        sample_idx += 10  # Move to next sample