"""Revisions and feature names that define the reproducible experiment."""

MODEL_ID = "lerobot/pi05_libero_finetuned_v044"
MODEL_REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
DATASET_ID = "lerobot/libero_plus"
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
LEROBOT_REVISION = "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
TRANSFORMERS_REVISION = "dcddb970176382c0fcf4521b0c0e6fc15894dfe0"
OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
PARC_REVISION = "eb8a063cf1d69615754b0cbb31b0a9162621ec9b"
TOKENIZER_ID = "google/paligemma-3b-pt-224"
TOKENIZER_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"

FRONT_DATASET_KEY = "observation.images.front"
WRIST_DATASET_KEY = "observation.images.wrist"
FRONT_POLICY_KEY = "observation.images.image"
WRIST_POLICY_KEY = "observation.images.image2"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

RENAME_MAP = {
    FRONT_DATASET_KEY: FRONT_POLICY_KEY,
    WRIST_DATASET_KEY: WRIST_POLICY_KEY,
}
