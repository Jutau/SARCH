#!/usr/bin/env python
"""
transformer2.py

A state-of-the-art Transformer model using advanced optimization techniques,
including mixed precision, XLA, distributed training, dynamic expert selection,
and optional TF-MOT (TensorFlow Model Optimization Toolkit) integration for pruning
and quantization.

Usage:
    Run the script directly for a demo of chain-of-thought reasoning and self-consistency sampling.
    Example:
        python transformer2.py

Requirements:
    - TensorFlow 2.x (with GPU support recommended)
    - transformers
    - (Optional) tensorflow_model_optimization

Note: This module is designed for experimentation. Feel free to modify and extend it.
"""

import tensorflow as tf
import numpy as np
from transformers import TFGPT2LMHeadModel, AutoTokenizer
import logging
import os

# Enable XLA (Accelerated Linear Algebra) and mixed precision.
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import TF-MOT for pruning and quantization.
try:
    import tensorflow_model_optimization as tfmot
    TF_MOT_AVAILABLE = True
    logger.info("TF-MOT enabled: Pruning & Quantization-Aware Training available.")
except ModuleNotFoundError:
    TF_MOT_AVAILABLE = False
    logger.warning("TF-MOT not found: Pruning & QAT are disabled.")

# Placeholder for efficient attention (to be implemented as needed).
def efficient_self_attention(x):
    # Stub for replacing standard attention with an efficient alternative (e.g., Performer).
    return x

class Transformer2(tf.keras.Model):
    def __init__(self, model_name: str = "gpt2", num_experts: int = 10) -> None:
        """
        Initialize Transformer² with a pre-trained GPT-2 LM Head model and dynamic expert vectors.
        """
        super(Transformer2, self).__init__()
        logger.info(f"Initializing Transformer2 with {model_name} and {num_experts} experts.")

        # Load GPT-2 with language modeling head.
        self.base_model = TFGPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.base_model.config.hidden_size
        self.vocab_size = self.base_model.config.vocab_size

        # SVD decomposition on the first attention weight for dynamic weight modulation.
        sample_weight = self.base_model.transformer.h[0].attn.c_attn.weight.numpy()
        U, S, V = self._svd_decomposition(sample_weight)
        self.U = tf.constant(U, dtype=tf.float32)
        self.S = tf.constant(S, dtype=tf.float32)
        self.V = tf.constant(V, dtype=tf.float32)

        # Expert vectors (dynamic modulators).
        self.num_experts = num_experts
        self.expert_vectors = tf.Variable(
            tf.random.normal([num_experts, self.S.shape[0]], stddev=0.01, dtype=tf.float32),
            trainable=True
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Fast adapter: now outputs dimension = vocab_size for logits.
        if TF_MOT_AVAILABLE:
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.3, final_sparsity=0.7, begin_step=0, end_step=1000
            )
            self.fast_adapter = tfmot.sparsity.keras.prune_low_magnitude(
                tf.keras.layers.Dense(self.vocab_size, activation='relu', dtype=tf.float32),
                pruning_schedule=pruning_schedule
            )
        else:
            self.fast_adapter = tf.keras.layers.Dense(self.vocab_size, activation='relu', dtype=tf.float32)

        # Expert selector: small MLP to dynamically select an expert based on task embedding.
        self.expert_selector = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', dtype=tf.float32),
            tf.keras.layers.Dense(num_experts, activation='softmax', dtype=tf.float32)
        ])

    def _svd_decomposition(self, weight_matrix: np.ndarray) -> tuple:
        """Compute SVD (u, s, v) with minimal memory footprint."""
        s, u, v = tf.linalg.svd(weight_matrix, full_matrices=False)
        return u.numpy(), s.numpy(), v.numpy()

    def _modulate_weights(self, expert_vector: tf.Tensor) -> tf.Tensor:
        """Modulate the singular values with the expert vector and reconstruct weights."""
        adapted_S = self.S * (1 + expert_vector)
        adapted_weight = tf.matmul(self.U,
                                   tf.matmul(tf.linalg.diag(adapted_S),
                                             tf.transpose(self.V)))
        return adapted_weight

    @tf.function(experimental_compile=True)
    def adapt_to_task(self, task_input: dict, expert_idx: int = None) -> tf.Tensor:
        """
        Two-pass adaptation with dynamic expert selection:
          1. First pass: run the base model with output_hidden_states=True to obtain a task embedding.
          2. Choose an expert and modulate weights.
          3. Second pass: run the model with modulated weights and apply the fast adapter.
        """
        # First pass: obtain hidden states.
        first_pass_output = self.base_model(**task_input, return_dict=True, output_hidden_states=True)
        hidden_states = first_pass_output.hidden_states  # Tuple of hidden states.
        last_hidden_state = hidden_states[-1]
        # Optionally, apply efficient attention.
        last_hidden_state = efficient_self_attention(last_hidden_state)
        task_embedding = tf.reduce_mean(last_hidden_state, axis=1)

        # Dynamic expert selection if not specified.
        if expert_idx is None:
            expert_probs = self.expert_selector(task_embedding)
            expert_idx = tf.argmax(expert_probs, axis=-1)[0]

        expert_vector = self.expert_vectors[expert_idx]
        adapted_weight = self._modulate_weights(expert_vector)

        # Backup original weights and assign modulated weights.
        original_weight = self.base_model.transformer.h[0].attn.c_attn.weight.read_value()
        self.base_model.transformer.h[0].attn.c_attn.weight.assign(adapted_weight)

        # Second pass: inference with modulated weights.
        second_pass = self.base_model(**task_input, return_dict=True).logits
        # Restore original weights.
        self.base_model.transformer.h[0].attn.c_attn.weight.assign(original_weight)

        # Apply fast adapter with a residual connection.
        fast_output = self.fast_adapter(second_pass)
        # Cast fast_output to match second_pass dtype.
        fast_output = tf.cast(fast_output, second_pass.dtype)
        return second_pass + fast_output

    def chain_of_thought(self, input_prompt: str, max_steps: int = 3) -> str:
        """Generate step-by-step reasoning (chain-of-thought) using real decoding."""
        steps = []
        current_prompt = input_prompt
        for step in range(max_steps):
            step_prompt = f"Step {step + 1}: {current_prompt}"
            inputs = self.tokenizer(step_prompt, return_tensors="tf", padding=True, truncation=True)
            step_output = self.call(inputs)
            step_text = self._decode_output(step_output)
            steps.append(step_text)
            current_prompt = f"{current_prompt}\n{step_text}"
        return steps[-1]

    def self_consistency(self, input_prompt: str, num_samples: int = 3, max_steps: int = 3) -> str:
        """Parallel chain-of-thought sampling and selecting the most frequent answer."""
        from concurrent.futures import ThreadPoolExecutor

        def sample_cot(_):
            return self.chain_of_thought(input_prompt, max_steps)

        with ThreadPoolExecutor(max_workers=num_samples) as executor:
            outputs = list(executor.map(sample_cot, range(num_samples)))
        return max(set(outputs), key=outputs.count)

    def _decode_output(self, output: tf.Tensor) -> str:
        """Decode output tokens using GPT-2's tokenizer."""
        token_ids = tf.argmax(output, axis=-1)
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)

    @tf.function(experimental_compile=True)
    def call(self, inputs: dict = None, training: bool = False, **kwargs) -> tf.Tensor:
        """Forward pass with proper handling of input dictionaries."""
        if inputs is not None:
            return self.adapt_to_task(inputs)
        task_input = {key: value for key, value in kwargs.items() if key in ["input_ids", "attention_mask"]}
        return self.adapt_to_task(task_input)

    def convert_to_tflite(self, representative_data_gen=None) -> bytes:
        """Convert the model to TFLite with quantization optimizations."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data_gen:
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        return converter.convert()

    def convert_to_trt(self, precision_mode="FP16") -> tf.keras.Model:
        """
        Convert the model for high-performance inference using TensorRT.
        precision_mode: 'FP16' or 'INT8'.
        """
        params = tf.experimental.tensorrt.ConversionParams(precision_mode=precision_mode)
        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=self.save_model_temp(),
            conversion_params=params
        )
        converter.convert()
        converter.build(input_fn=lambda: [(
            tf.random.uniform([1, 16], maxval=50257, dtype=tf.int32),
            tf.random.uniform([1, 16], maxval=1, dtype=tf.int32)
        )])
        trt_model_dir = "trt_model"
        converter.save(trt_model_dir)
        logger.info(f"TRT model saved to {trt_model_dir}.")
        return tf.saved_model.load(trt_model_dir)

    def save_model_temp(self) -> str:
        """Save the model temporarily for TRT conversion."""
        tmp_dir = "./temp_saved_model"
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
        self.save(tmp_dir, include_optimizer=False)
        return tmp_dir

    def apply_quantization(self) -> tf.keras.Model:
        """Apply Quantization-Aware Training (QAT) if TF-MOT is available."""
        if TF_MOT_AVAILABLE:
            quantize_model = tfmot.quantization.keras.quantize_model
            q_aware_model = quantize_model(self)
            logger.info("Quantization-aware model created.")
            return q_aware_model
        logger.warning("TF-MOT not available; skipping QAT.")
        return self


# Distributed training/inference with MirroredStrategy if multiple GPUs are available.
strategy = tf.distribute.MirroredStrategy()

def create_dataset(texts, tokenizer, batch_size=1):
    """Create an asynchronous input pipeline with caching and prefetching."""
    def gen():
        for text in texts:
            yield tokenizer(text, return_tensors="tf", padding="max_length", truncation=True)
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "input_ids": tf.TensorSpec(shape=(1, None), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(1, None), dtype=tf.int32)
        }
    )
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE).batch(batch_size)
    return dataset


if __name__ == "__main__":
    import time

    sample_input = "Solve this math problem: 2 + 2 = ?"

    # Initialize the model within the distributed strategy scope.
    with strategy.scope():
        start_time = time.time()
        model = Transformer2()
        init_time = time.time()
        logger.info(f"Initialization: {init_time - start_time:.2f} seconds")

    tokenized_input = model.tokenizer(sample_input, return_tensors="tf", padding=True, truncation=True)

    # Single pass inference.
    output = model.call(inputs=tokenized_input)
    call_time = time.time()
    logger.info(f"Single Pass: {call_time - init_time:.2f} seconds")

    # Self-consistency sampling.
    sc_output = model.self_consistency(sample_input)
    sc_time = time.time()
    logger.info(f"Self-Consistency: {sc_time - call_time:.2f} seconds, Answer: {sc_output}")

    # Chain-of-thought reasoning.
    cot_output = model.chain_of_thought(sample_input)
    cot_time = time.time()
    logger.info(f"Chain-of-Thought: {cot_time - sc_time:.2f} seconds, Answer: {cot_output}")

    # Optionally: Convert to TFLite or TensorRT for deployment.
    # tflite_model = model.convert_to_tflite()
    # trt_model = model.convert_to_trt(precision_mode="FP16")
