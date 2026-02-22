import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Using basic neural network implementation (can replace with TensorFlow/PyTorch)
class DenseLayer:
    """Fully connected neural network layer with Gradient Clipping"""
    
    def __init__(self, input_size, output_size, activation='relu'):
        # He Initialization (better for ReLU networks)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        
    def forward(self, X):
        """Forward pass"""
        self.input = X
        self.z = np.dot(X, self.weights) + self.bias
        
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-np.clip(self.z, -500, 500)))
        elif self.activation == 'softmax':
            # Numerical stability: subtract max before exp
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif self.activation == 'tanh':
            self.output = np.tanh(self.z)
        else:  # linear
            self.output = self.z
            
        return self.output
    
    def backward(self, grad_output, learning_rate=0.001):
        """Backward pass with GRADIENT CLIPPING"""
        if self.activation == 'relu':
            grad_activation = (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            grad_activation = self.output * (1 - self.output)
        elif self.activation == 'tanh':
            grad_activation = 1 - self.output ** 2
        elif self.activation == 'softmax':
            grad_activation = 1  # Handled in loss function
        else:
            grad_activation = 1
            
        if self.activation != 'softmax':
            grad_output = grad_output * grad_activation
            
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        
        # GRADIENT CLIPPING - Prevents NaN from exploding gradients
        grad_weights = np.clip(grad_weights, -1.0, 1.0)
        grad_bias = np.clip(grad_bias, -1.0, 1.0)
        
        # Update weights
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input


class DeepNeuralNetwork:
    """Deep Neural Network for classification"""
    
    def __init__(self, layer_sizes, activations):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            self.layers.append(layer)
    
    def forward(self, X):
        """Forward propagation through all layers"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true, learning_rate=0.001):
        """Backward propagation through all layers"""
        # Compute gradient of loss w.r.t. output
        grad = self.layers[-1].output - y_true
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001, class_weights=None):
        """Train the neural network with optional class weights"""
        n_samples = X_train.shape[0]
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        # Auto-calculate class weights if not provided
        if class_weights is None:
            # Get class labels from one-hot encoded y_train
            class_labels = np.argmax(y_train, axis=1)
            n_classes = y_train.shape[1]
            class_weights = np.zeros(n_classes)
            
            for i in range(n_classes):
                n_class_i = np.sum(class_labels == i)
                if n_class_i > 0:
                    # Calculate base weight
                    base_weight = n_samples / (n_classes * n_class_i)
                    # MODERATE amplification with STRICT cap
                    if base_weight > 1.0:  # Minority class
                        amplified = base_weight ** 1.2  # Gentle amplification
                        class_weights[i] = min(amplified, 3.0)  # Cap at 3x (was 10x)
                    else:  # Majority class
                        class_weights[i] = max(base_weight ** 0.5, 0.5)  # Floor at 0.5
                else:
                    class_weights[i] = 1.0
            
            print(f"\n  ✓ Auto-calculated class weights (moderate): {class_weights}")
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Compute weighted loss
                sample_weights = np.sum(y_batch * class_weights, axis=1, keepdims=True)
                loss = -np.mean(sample_weights * y_batch * np.log(output + 1e-8))
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass (gradient scaled by class weights)
                self.backward(y_batch * sample_weights, learning_rate)
            
            # Calculate metrics
            train_loss = epoch_loss / n_batches
            train_acc = self.evaluate(X_train, y_train)
            val_loss = self.compute_loss(X_val, y_val)
            val_acc = self.evaluate(X_val, y_val)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f} - "
                      f"Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """Predict class labels"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.forward(X)
    
    def evaluate(self, X, y):
        """Evaluate accuracy"""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return accuracy_score(true_labels, predictions)
    
    def compute_loss(self, X, y):
        """Compute cross-entropy loss"""
        output = self.forward(X)
        return -np.mean(y * np.log(output + 1e-8))


# ==================== FLIGHT DELAY PREDICTION MODEL ====================
class FlightDelayPredictor:
    """DNN model for predicting flight delays"""
    
    def __init__(self, input_size):
        # Network architecture: input -> 128 -> 64 -> 32 -> 2 (delayed/on-time)
        layer_sizes = [input_size, 128, 64, 32, 2]
        activations = ['relu', 'relu', 'relu', 'softmax']
        self.model = DeepNeuralNetwork(layer_sizes, activations)
        self.scaler = StandardScaler()
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the delay prediction model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert labels to one-hot
        y_train_onehot = np.zeros((len(y_train), 2))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        y_val_onehot = np.zeros((len(y_val), 2))
        y_val_onehot[np.arange(len(y_val)), y_val] = 1
        
        print("Training Flight Delay Prediction Model...")
        print("-" * 70)
        history = self.model.train(X_train_scaled, y_train_onehot, 
                                   X_val_scaled, y_val_onehot, 
                                   epochs=epochs, batch_size=64, learning_rate=0.001)
        return history
    
    def predict(self, X):
        """Predict delay status"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict delay probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


# ==================== FLIGHT REROUTING MODEL ====================
class FlightReroutingModel:
    """DNN model for intelligent flight rerouting decisions"""
    
    def __init__(self, input_size, n_routes):
        # Network architecture for rerouting decision
        # Input: flight features + delay prediction + route alternatives
        layer_sizes = [input_size, 256, 128, 64, n_routes]
        activations = ['relu', 'relu', 'relu', 'softmax']
        self.model = DeepNeuralNetwork(layer_sizes, activations)
        self.scaler = StandardScaler()
        self.n_routes = n_routes
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the rerouting model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert labels to one-hot
        y_train_onehot = np.zeros((len(y_train), self.n_routes))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        y_val_onehot = np.zeros((len(y_val), self.n_routes))
        y_val_onehot[np.arange(len(y_val)), y_val] = 1
        
        print("\nTraining Flight Rerouting Model...")
        print("-" * 70)
        history = self.model.train(X_train_scaled, y_train_onehot,
                                   X_val_scaled, y_val_onehot,
                                   epochs=epochs, batch_size=64, learning_rate=0.001)
        return history
    
    def predict(self, X):
        """Predict best route"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict route probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


# ==================== DATA GENERATION ====================
def generate_flight_data(n_samples=10000):
    """Generate synthetic flight data"""
    np.random.seed(42)
    
    data = {
        # Flight characteristics
        'origin_lat': np.random.uniform(25, 50, n_samples),
        'origin_lon': np.random.uniform(-125, -70, n_samples),
        'dest_lat': np.random.uniform(25, 50, n_samples),
        'dest_lon': np.random.uniform(-125, -70, n_samples),
        'distance': np.random.uniform(200, 3000, n_samples),
        'scheduled_time': np.random.uniform(1, 8, n_samples),  # hours
        
        # Time features
        'month': np.random.randint(1, 13, n_samples),
        'day_of_week': np.random.randint(1, 8, n_samples),
        'departure_hour': np.random.randint(5, 23, n_samples),
        
        # Operational features
        'aircraft_age': np.random.uniform(0, 25, n_samples),
        'carrier_reliability': np.random.uniform(0.7, 0.98, n_samples),
        'airport_congestion': np.random.uniform(0, 1, n_samples),
        
        # Weather features
        'origin_weather': np.random.uniform(0, 10, n_samples),
        'dest_weather': np.random.uniform(0, 10, n_samples),
        'route_weather': np.random.uniform(0, 10, n_samples),
        
        # Historical data
        'prev_delay': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'carrier_avg_delay': np.random.uniform(10, 60, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate delay labels (0: on-time, 1: delayed)
    delay_prob = (
        0.05 +
        0.15 * (df['month'].isin([6, 7, 12])) +
        0.10 * (df['day_of_week'].isin([5, 7])) +
        0.15 * (df['departure_hour'].isin([6, 7, 17, 18])) +
        0.10 * (df['origin_weather'] > 7) +
        0.10 * (df['dest_weather'] > 7) +
        0.15 * (df['airport_congestion'] > 0.7) +
        0.20 * df['prev_delay'] +
        0.05 * (df['aircraft_age'] > 15)
    )
    
    df['delay'] = (np.random.random(n_samples) < delay_prob).astype(int)
    
    return df


def generate_rerouting_data(flight_df, delay_predictions):
    """Generate rerouting dataset based on delay predictions"""
    n_samples = len(flight_df)
    
    # Alternative route features
    rerouting_data = {
        # Original flight features
        'distance': flight_df['distance'].values,
        'scheduled_time': flight_df['scheduled_time'].values,
        'airport_congestion': flight_df['airport_congestion'].values,
        'origin_weather': flight_df['origin_weather'].values,
        'dest_weather': flight_df['dest_weather'].values,
        
        # Delay prediction features
        'delay_probability': delay_predictions[:, 1],  # Probability of delay
        'predicted_delay': np.argmax(delay_predictions, axis=1),
        
        # Alternative routes (3 options)
        'route1_distance': flight_df['distance'].values * np.random.uniform(0.95, 1.05, n_samples),
        'route1_time': flight_df['scheduled_time'].values * np.random.uniform(0.9, 1.1, n_samples),
        'route1_weather': np.random.uniform(0, 10, n_samples),
        'route1_cost': np.random.uniform(500, 2000, n_samples),
        
        'route2_distance': flight_df['distance'].values * np.random.uniform(1.0, 1.15, n_samples),
        'route2_time': flight_df['scheduled_time'].values * np.random.uniform(0.85, 1.0, n_samples),
        'route2_weather': np.random.uniform(0, 10, n_samples),
        'route2_cost': np.random.uniform(600, 2200, n_samples),
        
        'route3_distance': flight_df['distance'].values * np.random.uniform(1.1, 1.25, n_samples),
        'route3_time': flight_df['scheduled_time'].values * np.random.uniform(0.8, 0.95, n_samples),
        'route3_weather': np.random.uniform(0, 10, n_samples),
        'route3_cost': np.random.uniform(700, 2500, n_samples),
    }
    
    reroute_df = pd.DataFrame(rerouting_data)
    
    # Generate optimal route labels (0: original, 1: route1, 2: route2, 3: route3)
    # Decision logic based on delay prediction and route characteristics
    route_scores = np.zeros((n_samples, 4))
    
    # Original route score
    route_scores[:, 0] = (
        100 - 50 * reroute_df['delay_probability'].values -
        2 * reroute_df['origin_weather'].values -
        2 * reroute_df['dest_weather'].values
    )
    
    # Alternative route scores
    for i in range(1, 4):
        route_scores[:, i] = (
            100 - 
            0.5 * (reroute_df[f'route{i}_weather'].values) -
            0.01 * (reroute_df[f'route{i}_cost'].values) +
            10 * (reroute_df[f'route{i}_time'].values < reroute_df['scheduled_time'].values)
        )
    
    # Add some randomness
    route_scores += np.random.randn(n_samples, 4) * 5
    
    optimal_routes = np.argmax(route_scores, axis=1)
    
    return reroute_df, optimal_routes


# ==================== MAIN EXECUTION ====================
def main():
    print("=" * 70)
    print("DNN FLIGHT REROUTING MODEL BASED ON DELAY PREDICTIONS")
    print("=" * 70)
    
    # Step 1: Generate flight data
    print("\n1. Generating flight dataset...")
    flight_df = generate_flight_data(n_samples=10000)
    print(f"Dataset shape: {flight_df.shape}")
    print(f"Delay distribution: {flight_df['delay'].value_counts().to_dict()}")
    
    # Step 2: Prepare data for delay prediction
    print("\n2. Preparing delay prediction dataset...")
    delay_features = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon', 'distance',
                     'scheduled_time', 'month', 'day_of_week', 'departure_hour',
                     'aircraft_age', 'carrier_reliability', 'airport_congestion',
                     'origin_weather', 'dest_weather', 'route_weather',
                     'prev_delay', 'carrier_avg_delay']
    
    X_delay = flight_df[delay_features].values
    y_delay = flight_df['delay'].values
    
    # Split delay data
    X_delay_train, X_delay_temp, y_delay_train, y_delay_temp = train_test_split(
        X_delay, y_delay, test_size=0.3, random_state=42, stratify=y_delay
    )
    X_delay_val, X_delay_test, y_delay_val, y_delay_test = train_test_split(
        X_delay_temp, y_delay_temp, test_size=0.5, random_state=42, stratify=y_delay_temp
    )
    
    print(f"Delay Train: {X_delay_train.shape[0]}, Val: {X_delay_val.shape[0]}, Test: {X_delay_test.shape[0]}")
    
    # Step 3: Train delay prediction model
    print("\n3. Training Delay Prediction DNN...")
    print("=" * 70)
    delay_predictor = FlightDelayPredictor(input_size=len(delay_features))
    delay_history = delay_predictor.fit(X_delay_train, y_delay_train,
                                       X_delay_val, y_delay_val,
                                       epochs=50)
    
    # Evaluate delay predictor
    print("\n4. Evaluating Delay Prediction Model...")
    print("-" * 70)
    y_delay_pred = delay_predictor.predict(X_delay_test)
    delay_accuracy = accuracy_score(y_delay_test, y_delay_pred)
    print(f"Delay Prediction Test Accuracy: {delay_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_delay_test, y_delay_pred,
                              target_names=['On-time', 'Delayed']))
    
    # Step 5: Generate delay predictions for all data
    print("\n5. Generating delay predictions for rerouting...")
    delay_probs = delay_predictor.predict_proba(X_delay)
    
    # Step 6: Generate rerouting dataset
    print("\n6. Generating rerouting dataset...")
    reroute_df, optimal_routes = generate_rerouting_data(flight_df, delay_probs)
    print(f"Rerouting dataset shape: {reroute_df.shape}")
    print(f"Route distribution: {pd.Series(optimal_routes).value_counts().to_dict()}")
    
    # Step 7: Prepare rerouting data
    X_reroute = reroute_df.values
    y_reroute = optimal_routes
    
    # Split rerouting data
    X_reroute_train, X_reroute_temp, y_reroute_train, y_reroute_temp = train_test_split(
        X_reroute, y_reroute, test_size=0.3, random_state=42
    )
    X_reroute_val, X_reroute_test, y_reroute_val, y_reroute_test = train_test_split(
        X_reroute_temp, y_reroute_temp, test_size=0.5, random_state=42
    )
    
    print(f"Reroute Train: {X_reroute_train.shape[0]}, Val: {X_reroute_val.shape[0]}, Test: {X_reroute_test.shape[0]}")
    
    # Step 8: Train rerouting model
    print("\n7. Training Rerouting DNN...")
    print("=" * 70)
    reroute_model = FlightReroutingModel(input_size=X_reroute.shape[1], n_routes=4)
    reroute_history = reroute_model.fit(X_reroute_train, y_reroute_train,
                                       X_reroute_val, y_reroute_val,
                                       epochs=50)
    
    # Step 9: Evaluate rerouting model
    print("\n8. Evaluating Rerouting Model...")
    print("-" * 70)
    y_reroute_pred = reroute_model.predict(X_reroute_test)
    reroute_accuracy = accuracy_score(y_reroute_test, y_reroute_pred)
    print(f"Rerouting Test Accuracy: {reroute_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_reroute_test, y_reroute_pred,
                              target_names=['Original', 'Route-1', 'Route-2', 'Route-3']))
    
    # Step 10: Demonstrate end-to-end pipeline
    print("\n9. End-to-End Pipeline Demonstration...")
    print("-" * 70)
    
    # Take a sample flight
    sample_idx = 0
    sample_flight = X_delay_test[sample_idx:sample_idx+1]
    
    # Predict delay
    delay_prob = delay_predictor.predict_proba(sample_flight)
    delay_pred = np.argmax(delay_prob)
    
    print(f"\nSample Flight Analysis:")
    print(f"Delay Probability: {delay_prob[0, 1]:.4f}")
    print(f"Predicted Status: {'DELAYED' if delay_pred == 1 else 'ON-TIME'}")
    
    # If delayed, suggest reroute
    if delay_prob[0, 1] > 0.5:
        sample_reroute = X_reroute_test[sample_idx:sample_idx+1]
        route_probs = reroute_model.predict_proba(sample_reroute)
        best_route = np.argmax(route_probs)
        
        route_names = ['Original Route', 'Alternative Route 1', 
                      'Alternative Route 2', 'Alternative Route 3']
        
        print(f"\nREROUTING RECOMMENDED!")
        print(f"Suggested Route: {route_names[best_route]}")
        print(f"Confidence: {route_probs[0, best_route]:.4f}")
        print("\nRoute Probabilities:")
        for i, name in enumerate(route_names):
            print(f"  {name}: {route_probs[0, i]:.4f}")
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
  