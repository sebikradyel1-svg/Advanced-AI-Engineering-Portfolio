import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Setare seed pentru reproducibilitate
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# 1. COMPONENTE TRANSFORMER
# ============================================================================

class PositionalEncoding(layers.Layer):
    """
    Codificare Pozițională pentru a adăuga informații despre poziția în secvență.
    Folosește funcții sinusoidale pentru a genera encodinguri.
    """
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles
    
    def positional_encoding(self, sequence_length, d_model):
        angle_rads = self.get_angles(
            np.arange(sequence_length)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Aplicare sin la indici pari
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Aplicare cos la indici impari
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention Layer.
    Implementează mecanismul de atenție cu mai multe capete.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model trebuie să fie divizibil cu num_heads"
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Împarte ultimul dimension în (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_output = tf.matmul(attention_weights, v)
        
        # Concatenare heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(concat_attention)
        return output


class TransformerEncoderBlock(layers.Layer):
    """
    Bloc Encoder Transformer complet.
    Include Multi-Head Attention, Feed Forward Network și normalizări.
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training):
        # Multi-Head Attention cu conexiune reziduală
        attn_output = self.mha(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed Forward Network cu conexiune reziduală
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


# ============================================================================
# 2. MODEL TRANSFORMER ENCODER
# ============================================================================

def build_transformer_model(sequence_length, num_features, d_model=64, 
                           num_heads=4, num_encoder_layers=2, dff=128, 
                           dropout_rate=0.1):
    """
    Construiește modelul Transformer Encoder complet.
    """
    inputs = layers.Input(shape=(sequence_length, num_features))
    
    # Proiecție la dimensiunea d_model
    x = layers.Dense(d_model)(inputs)
    
    # Adăugare codificare pozițională
    x = PositionalEncoding(sequence_length, d_model)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Stive de Encoder Blocks
    for _ in range(num_encoder_layers):
        x = TransformerEncoderBlock(d_model, num_heads, dff, dropout_rate)(x, training=True)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers pentru predicție
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# ============================================================================
# 3. PREGĂTIREA DATELOR
# ============================================================================

def generate_synthetic_stock_data(n_samples=1000):
    """
    Generează date sintetice de prețuri acțiuni cu trend și sezonalitate.
    """
    time = np.arange(n_samples)
    
    # Trend liniar
    trend = 0.05 * time + 100
    
    # Sezonalitate
    seasonal = 10 * np.sin(2 * np.pi * time / 50)
    
    # Zgomot aleatoriu
    noise = np.random.normal(0, 5, n_samples)
    
    # Random walk component
    random_walk = np.cumsum(np.random.normal(0, 2, n_samples))
    
    price = trend + seasonal + noise + random_walk
    
    # Creează DataFrame
    df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'Close': price,
        'Volume': np.random.randint(1000000, 5000000, n_samples),
        'High': price + np.abs(np.random.normal(2, 1, n_samples)),
        'Low': price - np.abs(np.random.normal(2, 1, n_samples)),
        'Open': price + np.random.normal(0, 1, n_samples)
    })
    
    return df


def create_sequences(data, sequence_length):
    """
    Creează secvențe pentru antrenarea modelului de serie temporală.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])  # Predict Close price
    return np.array(X), np.array(y)


# ============================================================================
# 4. MAIN - ANTRENARE ȘI EVALUARE
# ============================================================================

def main():
    print("=" * 80)
    print("TRANSFORMER ENCODER PENTRU PREDICȚIA PREȚURILOR DE ACȚIUNI")
    print("=" * 80)
    
    # Generare date
    print("\n1. Generare date sintetice...")
    df = generate_synthetic_stock_data(n_samples=1000)
    print(f"   Date generate: {len(df)} înregistrări")
    print(f"   Features: {df.columns.tolist()}")
    
    # Pregătire features
    features = ['Close', 'Volume', 'High', 'Low', 'Open']
    data = df[features].values
    
    # Normalizare
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Parametri
    SEQUENCE_LENGTH = 30
    TRAIN_SPLIT = 0.8
    
    # Creare secvențe
    print(f"\n2. Creare secvențe (lungime: {SEQUENCE_LENGTH})...")
    X, y = create_sequences(data_scaled, SEQUENCE_LENGTH)
    print(f"   Shape X: {X.shape}")
    print(f"   Shape y: {y.shape}")
    
    # Split train/test
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n3. Split date:")
    print(f"   Train: {len(X_train)} sample")
    print(f"   Test: {len(X_test)} sample")
    
    # Construire model
    print(f"\n4. Construire model Transformer...")
    model = build_transformer_model(
        sequence_length=SEQUENCE_LENGTH,
        num_features=len(features),
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        dff=128,
        dropout_rate=0.1
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(model.summary())
    
    # Antrenare
    print(f"\n5. Antrenare model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Predicții
    print(f"\n6. Evaluare model...")
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()
    
    # Denormalizare pentru evaluare corectă
    y_train_actual = y_train * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    y_test_actual = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    y_pred_train_actual = y_pred_train * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    y_pred_test_actual = y_pred_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    
    # Metrici
    print("\n" + "=" * 80)
    print("REZULTATE EVALUARE")
    print("=" * 80)
    
    print("\nTRAIN SET:")
    print(f"  MSE:  {mean_squared_error(y_train_actual, y_pred_train_actual):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual)):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_train_actual, y_pred_train_actual):.4f}")
    print(f"  R²:   {r2_score(y_train_actual, y_pred_train_actual):.4f}")
    
    print("\nTEST SET:")
    print(f"  MSE:  {mean_squared_error(y_test_actual, y_pred_test_actual):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual)):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_test_actual, y_pred_test_actual):.4f}")
    print(f"  R²:   {r2_score(y_test_actual, y_pred_test_actual):.4f}")
    
    # Vizualizări
    print("\n7. Generare vizualizări...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss History
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Train Predictions
    axes[0, 1].plot(y_train_actual[:200], label='Actual', alpha=0.7)
    axes[0, 1].plot(y_pred_train_actual[:200], label='Predicted', alpha=0.7)
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].set_title('Train Set: Actual vs Predicted (primele 200 sample)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Test Predictions
    axes[1, 0].plot(y_test_actual, label='Actual', alpha=0.7, linewidth=2)
    axes[1, 0].plot(y_pred_test_actual, label='Predicted', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].set_title('Test Set: Actual vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Prediction Error Distribution
    errors = y_test_actual - y_pred_test_actual
    axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Error Distribution (Mean: {np.mean(errors):.2f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformer_stock_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print("Antrenare și evaluare finalizate cu succes!")
    print("Graficele au fost salvate în 'transformer_stock_prediction_results.png'")
    print("=" * 80)
    
    return model, history, scaler


if __name__ == "__main__":
    model, history, scaler = main()
