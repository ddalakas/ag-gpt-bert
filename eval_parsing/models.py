"""
Unlabelled dependency parsing head customised for the GPT-BERT architecture.    

Adapted from :
https://github.com/Heidelberg-NLP/ancient-language-models/blob/main/src/ancient-language-models/models.py
"""

from transformers.modeling_outputs import TokenClassifierOutput
import torch
from torch import nn
from modeling_ltgbert import LtgbertModel # GPT-BERT model implementation (despite LTG-BERT naming)

class DependencyGPTBertForTokenClassification(LtgbertModel):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config, **kwargs):
        super().__init__(config, add_mlm_layer=False, **kwargs)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.u_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_a_inv = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        return_dict=True,
        labels=None,
        **kwargs
    ):
        output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(
            input_ids, attention_mask
        )

        batch_size, seq_len, hidden_size = output.size()

        source = output.unsqueeze(2).expand(-1, -1, seq_len, -1)   
        target = output.unsqueeze(1).expand(-1, seq_len, -1, -1)
        function_g = self.v_a_inv(torch.tanh(self.u_a(source) + self.w_a(target))).squeeze(-1)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
            function_g = function_g.masked_fill(mask == 0, -1e4)

        p_head = nn.functional.log_softmax(function_g, dim=2)

        loss = None
        if labels is not None:
            loss_fct = nn.NLLLoss(ignore_index=-100)
            loss = loss_fct(p_head.view(-1, seq_len), labels.view(-1))

        if not return_dict:
            output = (p_head,)
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=p_head,
            hidden_states=None,
            attentions=None,
        )