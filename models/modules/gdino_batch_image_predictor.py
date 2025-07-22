import torch, bisect
from typing import List, Tuple
from transformers import AutoTokenizer
from Object_Centric_Local_Navigation.models.modules.utils import resize_and_normalize_tensor

class GDinoBatchImagePredictor:
    TRANSFORM_SIZE = 800
    TRANSFORM_MEAN = [0.485, 0.456, 0.406]
    TRANSFORM_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, gdino_model):
        self.model = gdino_model
        for p in self.model.parameters():  
            p.requires_grad = False
        self.model.eval()

    @ torch.no_grad()
    def predict(
            self, 
            batch_images: torch.Tensor,
            captions: List[str],
            box_threshold: float,
            text_threshold: float,
            remove_combined: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:

        captions = [self.preprocess_caption(cpn) for cpn in captions]

        batch_size = batch_images.shape[0]
        batch_images = resize_and_normalize_tensor(batch_images, self.TRANSFORM_SIZE, self.TRANSFORM_MEAN, self.TRANSFORM_STD)
        
        outputs = self.model(batch_images, captions=captions)
        prediction_logits = outputs["pred_logits"].sigmoid()
        prediction_boxes = outputs["pred_boxes"]

        mask = prediction_logits.max(dim=2)[0] > box_threshold
        logits_list = [prediction_logits[i][mask[i]] for i in range(batch_size)]
        boxes_list  = [prediction_boxes[i][mask[i]] for i in range(batch_size)]
        confidences_list = [logits_list[i].max(dim=1)[0] for i in range(batch_size)]

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(captions)
        phrases_list = []
        for batch in range(batch_size):
            if remove_combined:
                sep_idx = [i for i in range(len(tokenized['input_ids'][batch])) if tokenized['input_ids'][batch][i] in [101, 102, 1012]]
                
                phrases = []
                for logit in logits_list[batch]:
                    max_idx = logit.argmax()
                    insert_idx = bisect.bisect_left(sep_idx, max_idx)
                    right_idx = sep_idx[insert_idx]
                    left_idx = sep_idx[insert_idx - 1]
                    phrases.append(self.get_phrases_from_posmap(logit > text_threshold, tokenized['input_ids'][batch], tokenizer, left_idx, right_idx).replace('.', ''))
            else:
                phrases = [
                    self.get_phrases_from_posmap(logit > text_threshold, tokenized['input_ids'][batch], tokenizer).replace('.', '')
                    for logit
                    in logits_list[batch]
                ]
            phrases_list.append(phrases)
        
        return boxes_list, confidences_list, phrases_list

    @staticmethod
    def preprocess_caption(caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."
    
    @staticmethod
    def get_phrases_from_posmap(
        posmap: torch.BoolTensor, input_ids: List[int], tokenizer: AutoTokenizer, left_idx: int = 0, right_idx: int = 255
    ):
        assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
        if posmap.dim() == 1:
            posmap[0: left_idx + 1] = False
            posmap[right_idx:] = False
            non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
            token_ids = [input_ids[i] for i in non_zero_idx]
            return tokenizer.decode(token_ids)
        else:
            raise NotImplementedError("posmap must be 1-dim")
    
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype
