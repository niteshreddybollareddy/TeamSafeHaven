"""
SILICON-21: CryNet Architecture
- Problem-driven acoustic architecture.
- Three semantic pillars: Envelope, Spectral, and Temporal Rhythm.
- Gated Fusion logic for robust decision making.
- Auxiliary losses for representation learning (discarded at inference).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_model(input_shape=(64, 198, 1), num_classes=3, training=False):
    """
    CryNet: Semantic Triple-Flow Architecture
    training=True returns auxiliary outputs for loss calculation.
    """
    weight_decay = 1e-4
    reg = regularizers.l2(weight_decay)
    
    inputs = layers.Input(shape=input_shape, name="data")

    # ---------------------------------------------------------
    # PILLAR 1: ENVELOPE PILLAR (Sustained Modulation)
    # Focus: Rise/Fall times, persistence (1xK Convs)
    # ---------------------------------------------------------
    # Collapse frequency immediately to focus on global energy envelope
    env = layers.Conv2D(16, (64, 1), strides=(1, 1), padding='valid', activation=None)(inputs)
    env = layers.BatchNormalization()(env)
    env = layers.Activation('softplus')(env) # Keeps dynamics alive, doesn't clip low energy
    env = layers.Reshape((input_shape[1], 16))(env) # (Time, Channels)
    
    # Temporal dilations to see long context (500ms+) quickly
    e1 = layers.Conv1D(16, 5, padding='same', dilation_rate=1, kernel_regularizer=reg)(env)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.Activation('softplus')(e1) # Refined: Use softplus throughout to avoid zero-clipping
    
    e2 = layers.Conv1D(16, 5, padding='same', dilation_rate=2, kernel_regularizer=reg)(e1)
    e2 = layers.BatchNormalization()(e2)
    e3 = layers.Add()([e1, e2]) # Residual link
    e3 = layers.Activation('softplus')(e3)
    
    e4 = layers.Conv1D(32, 5, padding='same', dilation_rate=4, kernel_regularizer=reg)(e3)
    e4 = layers.BatchNormalization()(e4)
    e4 = layers.Activation('softplus')(e4)
    
    env_features = layers.GlobalAveragePooling1D()(e4)
    
    # Envelope Confidence & Aux Head
    env_conf = layers.Dense(1, activation='linear', name='env_logit')(env_features) # Logit for softmax
    env_aux = layers.Dense(1, activation='sigmoid', name='envelope_aux')(env_features)

    # ---------------------------------------------------------
    # PILLAR 2: SPECTRAL PILLAR (Timbre/Harmonics)
    # Focus: Harmonic spacing, bandwidth (Kx1 Convs)
    # ---------------------------------------------------------
    # Temporal smoothing before collapse to penalize transient spikes/speech
    spec_smoothed = layers.AveragePooling2D(pool_size=(1, 5), strides=(1, 2), padding='same')(inputs)
    
    # Collapse time to focus on stable spectral signature (198 // 2 = 99)
    spec = layers.Conv2D(16, (1, input_shape[1]//2), padding='valid', activation='relu')(spec_smoothed)
    spec = layers.Reshape((input_shape[0], 16))(spec) # (Freq, Channels)
    
    s1 = layers.Conv1D(16, 5, padding='same', kernel_regularizer=reg)(spec)
    s1 = layers.BatchNormalization()(s1)
    s1 = layers.Activation('relu')(s1)
    
    s2 = layers.Conv1D(32, 5, padding='same', kernel_regularizer=reg)(s1)
    s2 = layers.BatchNormalization()(s2)
    s2 = layers.Activation('relu')(s2)
    
    spec_features = layers.GlobalAveragePooling1D()(s2)
    
    # Spectral Confidence & Aux Head
    spec_conf = layers.Dense(1, activation='linear', name='spec_logit')(spec_features) # Logit for softmax
    spec_aux = layers.Dense(num_classes, activation='softmax', name='spectral_aux')(spec_features)

    # ---------------------------------------------------------
    # PILLAR 3: TEMPORAL PATTERN PILLAR (Rhythm/Repetition)
    # Focus: Long-term periodicity, Anti-Dog safeguard
    # ---------------------------------------------------------
    # Mixed frequency/time but aggressive time dilation
    r1 = layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(inputs)
    r1_t = layers.Reshape((-1, 8))(r1)
    
    # Asymmetric regularizer for separable convs:
    # Keeps temporal kernels conservative, allows fusion flexibility.
    reg_p = regularizers.l2(5e-5)

    # Depthwise temporal conv with massive dilation
    r2 = layers.SeparableConv1D(16, 7, padding='same', dilation_rate=4, 
                                depthwise_regularizer=reg, pointwise_regularizer=reg_p)(r1_t)
    r2 = layers.BatchNormalization()(r2)
    r2 = layers.Activation('relu')(r2)
    
    r3 = layers.SeparableConv1D(32, 7, padding='same', dilation_rate=8, 
                                depthwise_regularizer=reg, pointwise_regularizer=reg_p)(r2)
    r3 = layers.BatchNormalization()(r3)
    r3 = layers.Activation('relu')(r3)
    
    # REFINED: Avg + Std pooling to distinguish periodic baby cry from impulsive dog/speech
    t_avg = layers.GlobalAveragePooling1D()(r3)
    t_std = layers.Lambda(lambda x: tf.math.reduce_std(x, axis=1))(r3)
    temp_features = layers.Concatenate()([t_avg, t_std])
    
    # Temporal Confidence & Aux Head
    temp_conf = layers.Dense(1, activation='linear', name='temp_logit')(temp_features) # Logit for softmax
    temp_aux = layers.Dense(1, activation='sigmoid', name='temporal_aux')(temp_features)

    # ---------------------------------------------------------
    # COMPETITIVE GATED FUSION ENGINE
    # Forces the model to choose which pillar's evidence matters most
    # ---------------------------------------------------------
    logits = layers.Concatenate()([env_conf, spec_conf, temp_conf])
    gates = layers.Softmax(name='fusion_gates')(logits)
    
    # Split back into individual scalars
    g_env = layers.Lambda(lambda x: x[:, 0:1])(gates)
    g_spec = layers.Lambda(lambda x: x[:, 1:2])(gates)
    g_temp = layers.Lambda(lambda x: x[:, 2:3])(gates)
    
    f_env = layers.Multiply()([env_features, g_env])
    f_spec = layers.Multiply()([spec_features, g_spec])
    f_temp = layers.Multiply()([temp_features, g_temp])
    
    merged = layers.Concatenate()([f_env, f_spec, f_temp])
    
    # Final Classification Head
    x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(merged)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation='softmax', name='main_output')(x)

    # Final Model Construction
    if training:
        # Return main output and all auxiliary outputs for training
        model = models.Model(inputs=inputs, outputs=[output, env_aux, spec_aux, temp_aux])
    else:
        # Standard inference configuration
        model = models.Model(inputs=inputs, outputs=output)
        
    return model

if __name__ == '__main__':
    # Test inference build
    model_inf = create_model()
    model_inf.summary()
    
    # Test training build
    model_train = create_model(training=True)
    print("\nTraining Model Outputs:", [o.name for o in model_train.outputs])
