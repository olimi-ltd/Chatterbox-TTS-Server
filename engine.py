# File: engine.py
# Core TTS model loading and speech generation logic.

import gc
import logging
import random
import numpy as np
import torch
from typing import Optional, Tuple, Generator
from pathlib import Path
from safetensors.torch import load_file

import torch.nn.functional as F
from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
from transformers.generation.logits_process import (
    TopPLogitsWarper,
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)

# Defensive Turbo import - Turbo may not be available in older package versions
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    TURBO_AVAILABLE = True
except ImportError:
    ChatterboxTurboTTS = None
    TURBO_AVAILABLE = False

# Defensive Multilingual import
try:
    from chatterbox import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

    MULTILINGUAL_AVAILABLE = True
except ImportError:
    ChatterboxMultilingualTTS = None
    SUPPORTED_LANGUAGES = {}
    MULTILINGUAL_AVAILABLE = False

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# Log Turbo availability status at module load time
if TURBO_AVAILABLE:
    logger.info("ChatterboxTurboTTS is available in the installed chatterbox package.")
else:
    logger.info("ChatterboxTurboTTS not available in installed chatterbox package.")

# Log Multilingual availability status at module load time
if MULTILINGUAL_AVAILABLE:
    logger.info("ChatterboxMultilingualTTS is available in the installed chatterbox package.")
    logger.info(f"Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
else:
    logger.info("ChatterboxMultilingualTTS not available in installed chatterbox package.")

# Model selector whitelist - maps config values to model types
MODEL_SELECTOR_MAP = {
    # Original model selectors
    "chatterbox": "original",
    "original": "original",
    "resembleai/chatterbox": "original",
    # Turbo model selectors
    "chatterbox-turbo": "turbo",
    "turbo": "turbo",
    "resembleai/chatterbox-turbo": "turbo",
    # Multilingual model selectors
    "chatterbox-multilingual": "multilingual",
    "multilingual": "multilingual",
}

# Paralinguistic tags supported by Turbo model
TURBO_PARALINGUISTIC_TAGS = [
    "laugh",
    "chuckle",
    "sigh",
    "gasp",
    "cough",
    "clear throat",
    "sniff",
    "groan",
    "shush",
]

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)

# Track which model type is loaded
loaded_model_type: Optional[str] = None  # "original" or "turbo"
loaded_model_class_name: Optional[str] = None  # "ChatterboxTTS" or "ChatterboxTurboTTS"


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def _get_model_class(selector: str) -> tuple:
    """
    Determines which model class to use based on the config selector value.

    Args:
        selector: The value from config model.repo_id

    Returns:
        Tuple of (model_class, model_type_string)

    Raises:
        ImportError: If Turbo or Multilingual is selected but not available in the package
    """
    selector_normalized = selector.lower().strip()
    model_type = MODEL_SELECTOR_MAP.get(selector_normalized)

    if model_type == "turbo":
        if not TURBO_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxTurboTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Turbo model (ChatterboxTurboTTS)"
        )
        return ChatterboxTurboTTS, "turbo"

    if model_type == "multilingual":
        if not MULTILINGUAL_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxMultilingualTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Multilingual model (ChatterboxMultilingualTTS)"
        )
        return ChatterboxMultilingualTTS, "multilingual"

    if model_type == "original":
        logger.info(
            f"Model selector '{selector}' resolved to Original model (ChatterboxTTS)"
        )
        return ChatterboxTTS, "original"

    # Unknown selector - default to original with warning
    logger.warning(
        f"Unknown model selector '{selector}'. "
        f"Valid values: chatterbox, chatterbox-turbo, chatterbox-multilingual, original, turbo, multilingual, "
        f"ResembleAI/chatterbox, ResembleAI/chatterbox-turbo. "
        f"Defaulting to original ChatterboxTTS model."
    )
    return ChatterboxTTS, "original"


def get_model_info() -> dict:
    """
    Returns information about the currently loaded model.
    Used by the API to expose model details to the UI.

    Returns:
        Dictionary containing model information
    """
    return {
        "loaded": MODEL_LOADED,
        "type": loaded_model_type,  # "original", "turbo", or "multilingual"
        "class_name": loaded_model_class_name,
        "device": model_device,
        "sample_rate": chatterbox_model.sr if chatterbox_model else None,
        "supports_paralinguistic_tags": loaded_model_type == "turbo",
        "available_paralinguistic_tags": (
            TURBO_PARALINGUISTIC_TAGS if loaded_model_type == "turbo" else []
        ),
        "turbo_available_in_package": TURBO_AVAILABLE,
        "multilingual_available_in_package": MULTILINGUAL_AVAILABLE,
        "supports_multilingual": loaded_model_type == "multilingual",
        "supported_languages": (
            SUPPORTED_LANGUAGES if loaded_model_type == "multilingual" else {"en": "English"}
        ),
    }


def load_model() -> bool:
    """
    Loads the TTS model.
    This version supports both loading from Hugging Face Hub (via from_pretrained)
    and from a local checkpoint directory (via from_local).
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device
    global loaded_model_type, loaded_model_class_name

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # Determine processing device with robust CUDA detection and intelligent fallback
        device_setting = config_manager.get_string("tts_engine.device", "auto")

        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA and MPS not functional or not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with CUDA support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "mps":
            if _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS requested and functional. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "MPS was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with MPS support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
            elif _test_mps_functionality():
                resolved_device_str = "mps"
            else:
                resolved_device_str = "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")

        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")

        # Get the model selector from config
        model_selector = config_manager.get_string("model.repo_id", "chatterbox-turbo")

        # Check if user specified a local checkpoint path
        local_checkpoint = config_manager.get_string("model.local_checkpoint", "")

        logger.info(f"Model selector from config: '{model_selector}'")
        if local_checkpoint:
            logger.info(f"Local checkpoint path from config: '{local_checkpoint}'")

        try:
            # Determine which model class to use
            model_class, model_type = _get_model_class(model_selector)

            logger.info(
                f"Initializing {model_class.__name__} on device '{model_device}'..."
            )
            logger.info(f"Model type: {model_type}")
            if model_type == "turbo":
                logger.info(
                    f"Turbo model supports paralinguistic tags: {TURBO_PARALINGUISTIC_TAGS}"
                )

            # Load the model - check if loading from local checkpoint or from HuggingFace
            if local_checkpoint and Path(local_checkpoint).exists():
                checkpoint_path = Path(local_checkpoint)

                # Check if this is a full model directory or just fine-tuned weights
                has_full_model = (checkpoint_path / "ve.safetensors").exists()
                has_finetuned_weights = (checkpoint_path / "model.safetensors").exists()

                if has_full_model:
                    # Full model directory - load directly
                    logger.info(f"Loading complete model from local checkpoint: {local_checkpoint}")
                    chatterbox_model = model_class.from_local(
                        ckpt_dir=local_checkpoint,
                        device=model_device
                    )
                elif has_finetuned_weights:
                    # Only fine-tuned weights - load base model and replace T3 module
                    # This handles fine-tuned models with extended vocabularies (e.g., for Arabic)
                    logger.info(f"Loading base model from HuggingFace and applying fine-tuned weights from: {local_checkpoint}")
                    chatterbox_model = model_class.from_pretrained(device="cpu")  # Load to CPU first

                    # Load the fine-tuned T3 weights
                    finetuned_weights_path = checkpoint_path / "model.safetensors"
                    logger.info(f"Loading fine-tuned T3 weights from: {finetuned_weights_path}")
                    finetuned_state_dict = load_file(str(finetuned_weights_path))

                    # Detect vocabulary size from the text embedding layer
                    # The text_emb.weight tensor has shape [vocab_size, embed_dim]
                    vocab_size = None
                    for key in finetuned_state_dict.keys():
                        if 'text_emb.weight' in key:
                            vocab_size = finetuned_state_dict[key].shape[0]
                            logger.info(f"Detected vocabulary size from checkpoint: {vocab_size}")
                            break

                    if vocab_size is not None and vocab_size != chatterbox_model.t3.hp.text_tokens_dict_size:
                        # Extended vocabulary detected - need to create new T3 module
                        logger.info(f"Fine-tuned model has extended vocabulary (base: {chatterbox_model.t3.hp.text_tokens_dict_size}, fine-tuned: {vocab_size})")
                        logger.info("Creating new T3 module with extended vocabulary...")

                        # Import T3 class
                        try:
                            from chatterbox.models.t3.t3 import T3
                        except ImportError:
                            logger.error("Failed to import T3 class. Cannot create extended vocabulary T3.")
                            raise

                        # Get T3 config and update vocab size
                        t3_config = chatterbox_model.t3.hp
                        t3_config.text_tokens_dict_size = vocab_size

                        # Create new T3 with extended vocabulary
                        new_t3 = T3(hp=t3_config)

                        # Strip "t3." prefix from keys if present
                        cleaned_state_dict = {}
                        for key, value in finetuned_state_dict.items():
                            if key.startswith('t3.'):
                                cleaned_key = key[3:]  # Remove "t3." prefix
                                cleaned_state_dict[cleaned_key] = value
                            else:
                                cleaned_state_dict[key] = value

                        logger.info(f"Cleaned {len(finetuned_state_dict)} checkpoint keys (removed 't3.' prefix where present)")

                        # Load the fine-tuned weights into the new T3
                        new_t3.load_state_dict(cleaned_state_dict, strict=True)
                        logger.info("Successfully loaded fine-tuned weights into new T3 module with extended vocabulary")

                        # Replace the T3 module in the model
                        chatterbox_model.t3 = new_t3
                    else:
                        # No vocab size change, load weights directly
                        logger.info("Vocabulary size unchanged, loading weights into existing T3 module")
                        chatterbox_model.t3.load_state_dict(finetuned_state_dict, strict=False)
                        logger.info("Successfully loaded fine-tuned T3 weights into base model")

                    # Load extended tokenizer if present in checkpoint directory
                    custom_tokenizer_path = checkpoint_path / "tokenizer.json"
                    if custom_tokenizer_path.exists():
                        from chatterbox.models.tokenizers.tokenizer import EnTokenizer
                        logger.info(f"Loading custom tokenizer from: {custom_tokenizer_path}")
                        chatterbox_model.tokenizer = EnTokenizer(str(custom_tokenizer_path))
                        logger.info(f"Custom tokenizer loaded (vocab size: {len(chatterbox_model.tokenizer.tokenizer.get_vocab())})")
                    else:
                        logger.warning("No custom tokenizer.json found in checkpoint directory. Using base tokenizer - this may cause issues with extended vocabularies!")

                    # Move entire model to the target device
                    logger.info(f"Moving model to device: {model_device}")
                    chatterbox_model.t3 = chatterbox_model.t3.to(model_device)
                    chatterbox_model.s3gen = chatterbox_model.s3gen.to(model_device)
                    chatterbox_model.ve = chatterbox_model.ve.to(model_device)
                    chatterbox_model.device = model_device
                else:
                    logger.warning(
                        f"Local checkpoint path '{local_checkpoint}' exists but doesn't contain "
                        f"recognizable model files (ve.safetensors or model.safetensors). "
                        f"Falling back to loading from HuggingFace Hub."
                    )
                    chatterbox_model = model_class.from_pretrained(device=model_device)
            else:
                if local_checkpoint:
                    logger.warning(
                        f"Local checkpoint path '{local_checkpoint}' does not exist. "
                        f"Falling back to loading from HuggingFace Hub."
                    )
                # Load the model using from_pretrained - handles HuggingFace downloads automatically
                chatterbox_model = model_class.from_pretrained(device=model_device)

            # Store model metadata
            loaded_model_type = model_type
            loaded_model_class_name = model_class.__name__

            logger.info(f"Successfully loaded {model_class.__name__} on {model_device}")
            logger.info(f"Model sample rate: {chatterbox_model.sr} Hz")
        except ImportError as e_import:
            logger.error(
                f"Failed to load model due to import error: {e_import}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False
        except Exception as e_hf:
            logger.error(
                f"Failed to load model using from_pretrained: {e_hf}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded successfully on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz."
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the loaded TTS model.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness.
        cfg_weight: Classifier-Free Guidance weight.
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.
        language: Language code for multilingual model (e.g., 'en', 'it', 'de').

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    try:
        # Set seed globally if a specific seed value is provided and is non-zero.
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info(
                "Using default (potentially random) generation behavior as seed is 0."
            )

        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}, "
            f"language={language}"
        )

        # Call the core model's generate method
        # Multilingual model requires language_id parameter
        if loaded_model_type == "multilingual":
            wav_tensor = chatterbox_model.generate(
                text=text,
                language_id=language,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        else:
            wav_tensor = chatterbox_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        # The ChatterboxTTS.generate method already returns a CPU tensor.
        return wav_tensor, chatterbox_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None


# ---------------------------------------------------------------------------
# Token-level streaming helpers
# ---------------------------------------------------------------------------

def _decode_chunk(
    model,
    chunk_tokens: torch.Tensor,
    all_tokens: torch.Tensor,
    context_window: int,
    device,
) -> Optional[torch.Tensor]:
    """
    Decode a chunk of speech tokens to audio using S3Gen.

    Prepends up to context_window preceding tokens for continuity, trims
    the corresponding audio using a dynamic samples-per-token ratio, applies
    a 20 ms linear fade-in to smooth boundaries, and watermarks the result.
    Returns a (1, samples) float32 tensor, or None if output is empty.
    """
    FADE_SAMPLES = int(0.02 * S3GEN_SR)  # 20 ms at 24 kHz

    if all_tokens.numel() > 0:
        ctx = all_tokens[-context_window:] if all_tokens.numel() > context_window else all_tokens
        tokens_to_decode = torch.cat([ctx, chunk_tokens])
        context_length = ctx.numel()
    else:
        tokens_to_decode = chunk_tokens
        context_length = 0

    clean = drop_invalid_tokens(tokens_to_decode.to(device))
    if clean.numel() == 0:
        return None

    wav, _ = model.s3gen.inference(speech_tokens=clean, ref_dict=model.conds.gen)
    wav_np = wav.squeeze(0).detach().cpu().numpy()

    # Trim context using dynamic ratio (avoids hard-coded hop-size assumption)
    if context_length > 0:
        samples_per_token = len(wav_np) / len(clean)
        wav_np = wav_np[int(context_length * samples_per_token):]

    if len(wav_np) == 0:
        return None

    # Short linear fade-in to smooth chunk boundaries
    fade = min(FADE_SAMPLES, len(wav_np))
    if fade > 0:
        wav_np[:fade] *= np.linspace(0.0, 1.0, fade, dtype=wav_np.dtype)

    # Watermark each chunk (matches non-streaming generate() behaviour)
    wav_np = model.watermarker.apply_watermark(wav_np, sample_rate=model.sr)

    return torch.from_numpy(wav_np).unsqueeze(0)  # (1, samples)


def _stream_generate(
    model,
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    chunk_size: int = 25,
    context_window: int = 50,
    max_new_tokens: int = 1000,
    repetition_penalty: float = 1.2,
    min_p: float = 0.05,
    top_p: float = 0.95,
) -> Generator[Tuple[torch.Tensor, int], None, None]:
    """
    Token-level streaming generator. Replicates the T3 KV-cache loop from
    ChatterboxTTS.generate() but yields decoded audio as tokens are produced.
    """
    # --- Conditionals ---
    if audio_prompt_path:
        model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
    else:
        if model.conds is None:
            raise RuntimeError("No conditionals loaded. Provide audio_prompt_path or call prepare_conditionals first.")
        if exaggeration != model.conds.t3.emotion_adv[0, 0, 0].item():
            _cond = model.conds.t3
            model.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=model.device)

    # --- Tokenize (always duplicate for CFG batch) ---
    text = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(text).to(model.device)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # batch=2 for CFG
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    t3 = model.t3

    with torch.inference_mode():
        patched_model = T3HuggingfaceBackend(
            config=t3.cfg,
            llama=t3.tfmr,
            speech_enc=t3.speech_emb,
            speech_head=t3.speech_head,
            alignment_stream_analyzer=None,
        )

        initial_speech = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            speech_tokens=initial_speech,
            cfg_weight=cfg_weight,
        )

        device = embeds.device

        bos_token = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        generated_ids = bos_token.clone()

        top_p_proc     = TopPLogitsWarper(top_p=top_p)
        min_p_proc     = MinPLogitsWarper(min_p=min_p)
        rep_penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        output = patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

        # 1-D token accumulators for efficient slicing
        all_tokens = torch.empty(0, dtype=torch.long, device=device)
        chunk_buf   = torch.empty(0, dtype=torch.long, device=device)

        for i in range(max_new_tokens):
            logits_step = output.logits[:, -1, :]
            cond   = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            logits = cond + cfg_weight * (cond - uncond)

            ids_for_proc = generated_ids[:1, ...]
            if temperature != 1.0:
                logits = logits / temperature
            logits = rep_penalty_proc(ids_for_proc, logits)
            logits = min_p_proc(ids_for_proc, logits)
            logits = top_p_proc(ids_for_proc, logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # EOS: flush remaining buffer and stop
            if next_token.view(-1).item() == t3.hp.stop_speech_token:
                if chunk_buf.numel() > 0:
                    ctx_tokens = all_tokens[: all_tokens.numel() - chunk_buf.numel()]
                    audio = _decode_chunk(model, chunk_buf, ctx_tokens, context_window, device)
                    if audio is not None:
                        yield audio, model.sr
                break

            token_val  = next_token.view(-1)
            chunk_buf  = torch.cat([chunk_buf,  token_val])
            all_tokens = torch.cat([all_tokens, token_val])

            # Yield when chunk buffer is full
            if chunk_buf.numel() >= chunk_size:
                ctx_tokens = all_tokens[: all_tokens.numel() - chunk_buf.numel()]
                audio = _decode_chunk(model, chunk_buf, ctx_tokens, context_window, device)
                if audio is not None:
                    yield audio, model.sr
                chunk_buf = torch.empty(0, dtype=torch.long, device=device)

            # Advance KV-cache
            next_embed = t3.speech_emb(next_token) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            next_embed = torch.cat([next_embed, next_embed])
            output = patched_model(
                inputs_embeds=next_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values


def synthesize_stream(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    chunk_size: int = 25,
    context_window: int = 50,
) -> Generator[Tuple[torch.Tensor, int], None, None]:
    """
    Token-level streaming synthesis. Yields (audio_chunk_tensor, sample_rate) tuples
    as speech tokens are generated.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot stream audio.")
        return

    if seed != 0:
        logger.info(f"Applying seed for streaming generation: {seed}")
        set_seed(seed)

    logger.debug(
        f"Streaming synthesis: audio_prompt='{audio_prompt_path}', temp={temperature}, "
        f"exag={exaggeration}, cfg_weight={cfg_weight}, chunk_size={chunk_size}, "
        f"context_window={context_window}"
    )

    try:
        yield from _stream_generate(
            model=chatterbox_model,
            text=text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            chunk_size=chunk_size,
            context_window=context_window,
        )
    except Exception as e:
        logger.error(f"Error during streaming synthesis: {e}", exc_info=True)


def reload_model() -> bool:
    """
    Unloads the current model, clears GPU memory, and reloads the model
    based on the current configuration. Used for hot-swapping models
    without restarting the server process.

    Returns:
        bool: True if the new model loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name

    logger.info("Initiating model hot-swap/reload sequence...")

    # 1. Unload existing model
    if chatterbox_model is not None:
        logger.info("Unloading existing TTS model from memory...")
        del chatterbox_model
        chatterbox_model = None

    # 2. Reset state flags
    MODEL_LOADED = False
    loaded_model_type = None
    loaded_model_class_name = None

    # 3. Force Python Garbage Collection
    gc.collect()
    logger.info("Python garbage collection completed.")

    # 4. Clear GPU Cache (CUDA)
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()

    # 5. Clear GPU Cache (MPS - Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache.")
        except AttributeError:
            # Older PyTorch versions may not have mps.empty_cache()
            logger.debug(
                "torch.mps.empty_cache() not available in this PyTorch version."
            )

    # 6. Reload model from the (now updated) configuration
    logger.info("Memory cleared. Reloading model from updated config...")
    return load_model()


# --- End File: engine.py ---
