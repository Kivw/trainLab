import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, ChineseCLIPModel
from trainlab.builder import MODELS

@MODELS.register_module()
class TransformersModelWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import (
            AutoModel, 
            AutoModelForCausalLM, 
            AutoModelForSeq2SeqLM,
            AutoTokenizer, 
            AutoProcessor, 
            AutoFeatureExtractor,
            AutoImageProcessor
        )

        model_name = cfg.get("pretrained")
        model_type = cfg.get("model_type", "auto")  # auto / causal / seq2seq

        # ------------------------
        # 1️⃣ 加载模型
        # ------------------------
        if model_type == "causal":
            self.origin_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        elif model_type == "seq2seq":
            self.origin_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.origin_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # ------------------------
        # 2️⃣ 自动加载 Tokenizer / Processor
        # ------------------------
        self.tokenizer = None
        self.processor = None

        # 优先尝试 processor（多模态、音频类）
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print(f"[INFO] Loaded processor for {model_name}")
        except Exception:
            # 否则尝试 tokenizer（纯文本）
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                print(f"[INFO] Loaded tokenizer for {model_name}")
            except Exception:
                # 再尝试 feature extractor（图像类）
                try:
                    self.processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
                    print(f"[INFO] Loaded feature extractor for {model_name}")
                except Exception:
                    # 最后尝试 image processor（新版本图像模型）
                    try:
                        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
                        print(f"[INFO] Loaded image processor for {model_name}")
                    except Exception:
                        print(f"[WARNING] No tokenizer/processor found for {model_name}")

    # ------------------------
    # 3️⃣ Forward 封装
    # ------------------------
    def forward(self, *args, **kwargs):
        return self.origin_model(*args, **kwargs)
