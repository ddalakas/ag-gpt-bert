import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from .configuration_ltgbert import LtgbertConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import gelu_new
from transformers.modeling_outputs import (
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    BaseModelOutput,
    CausalLMOutput
)
from transformers.pytorch_utils import softmax_backward_data


class InPlaceSetSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_idx, x_val):
        full_tensor[x_idx] = x_val
        ctx.x_idx = x_idx
        ret = torch.Tensor().to(full_tensor.device)
        ret.set_(full_tensor[:x_idx + 1])
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.x_idx == 0:
            return None, None, None, grad_out[ctx.x_idx]
        else:
            return None, grad_out[:ctx.x_idx], None, grad_out[ctx.x_idx]


def apply_inplace_set(x_acc, x_idx, x_val):
    full_tensor, last_slice = x_acc
    new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val)
    return full_tensor, new_slice


class DWAModules(torch.nn.Module):
    def __init__(self, hidden_size, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(i + 2)) for i in range(n_blocks)])
        self.accumulator = None
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.alphas:
            module.data.zero_()
            module.data[-1] = 1.0

    def init_accumulator(self, x):
        self.accumulator = (torch.zeros((self.n_blocks + 1, *x.shape), device=x.device, dtype=x.dtype), None)
        self.accumulator = apply_inplace_set(self.accumulator, 0, x)

    def forward(self, x, block_idx):
        assert self.accumulator is not None, "`init_accumulator(x)` needs to be called first"
        self.accumulator = apply_inplace_set(
            self.accumulator,
            block_idx + 1,
            x
        )
        x = torch.tensordot(self.alphas[block_idx], self.accumulator[1], dims=1)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_layers = nn.ModuleList([Attention(config) for _ in range(config.num_hidden_layers)])
        self.mlp_layers = nn.ModuleList([FeedForward(config) for _ in range(config.num_hidden_layers)])
        self.dwa_modules = DWAModules(config.hidden_size, config.num_hidden_layers * 2)

        for i, layer in enumerate(self.mlp_layers):
            layer.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

    def forward(self, x, attention_mask, relative_embedding):
        hidden_states, attention_probs = [x], []

        self.dwa_modules.init_accumulator(x)
        for i, (attention_layer, mlp_layer) in enumerate(zip(self.attention_layers, self.mlp_layers)):
            attention_output, attention_p = attention_layer(x, attention_mask, relative_embedding)
            x = x + attention_output
            x = self.dwa_modules(x, block_idx=i * 2)

            x = x + mlp_layer(x)
            x = self.dwa_modules(x, block_idx=i * 2 + 1)

            hidden_states.append(x)
            attention_probs.append(attention_p)

        return hidden_states, attention_probs


class MaskClassifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )

    def forward(self, x, masked_lm_labels=None):
        if masked_lm_labels is not None:
            x = torch.index_select(x.flatten(0, 1), 0, torch.nonzero(masked_lm_labels.flatten() != -100).squeeze())
        x = self.nonlinearity(x)
        return x


# class EncoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attention = Attention(config)
#         self.mlp = FeedForward(config)

#     def forward(self, x, padding_mask, relative_embedding):
#         attention_output, attention_probs = self.attention(x, padding_mask, relative_embedding)
#         x = x + attention_output
#         x = x + self.mlp(x)
#         return x, attention_probs


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * gelu_new(gate)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(self, x):
        return self.mlp(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        self.dim = dim

        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        input_grad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return input_grad, None, None


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.in_proj_vg = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)

        position_indices = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1) \
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos).clamp(max=max_position - 1))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def forward(self, hidden_states, attention_mask, relative_embedding):
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        if self.position_indices.size(0) < query_len:
            position_indices = torch.arange(query_len, dtype=torch.long).unsqueeze(1) \
                - torch.arange(query_len, dtype=torch.long).unsqueeze(0)
            position_indices = self.make_log_bucket_position(position_indices, self.config.position_bucket_size, 512)
            position_indices = self.config.position_bucket_size - 1 + position_indices
            self.position_indices = position_indices.to(hidden_states.device)

        hidden_states = self.pre_layer_norm(hidden_states)

        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
        value, gate = self.in_proj_vg(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
        gate = F.gelu(gate)

        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        query_pos, key_pos = self.in_proj_qk(self.dropout(relative_embedding)).chunk(2, dim=-1)  # shape: [2T-1, D]
        query_pos = query_pos.view(-1, self.num_heads, self.head_size)  # shape: [2T-1, H, D]
        key_pos = key_pos.view(-1, self.num_heads, self.head_size)  # shape: [2T-1, H, D]

        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, query_len, self.head_size)

        attention_c_p = torch.einsum("bhqd,khd->bhqk", query, key_pos.squeeze(1) * self.scale)
        attention_p_c = torch.einsum("bhkd,qhd->bhqk", key * self.scale, query_pos.squeeze(1))

        position_indices = self.position_indices[:query_len, :key_len].expand(batch_size, self.num_heads, -1, -1)
        attention_c_p = attention_c_p.gather(3, position_indices)
        attention_p_c = attention_p_c.gather(2, position_indices)

        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(attention_c_p)
        attention_scores.add_(attention_p_c)

        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)

        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context = context * gate
        context = self.post_layer_norm(context)
        context = self.out_proj(context)
        context = self.dropout(context)

        return context, attention_probs.detach()


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings


#
# HuggingFace wrappers
#

class LtgbertPreTrainedModel(PreTrainedModel):
    config_class = LtgbertConfig
    supports_gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        raise NotImplementedError("Gradient checkpointing is not supported by this model")

    def _init_weights(self, module):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))

        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2*std, b=2*std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2*std, b=2*std)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


class LtgbertModel(LtgbertPreTrainedModel):
    def __init__(self, config, add_mlm_layer=False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embedding = Embedding(config)
        self.transformer = Encoder(config)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight) if add_mlm_layer else None


    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def get_contextualized_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        else:
            attention_mask = ~attention_mask.bool()

        if self.config.is_decoder:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) | torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device), 1).unsqueeze(0).unsqueeze(0)
        else:
            if len(attention_mask.size()) == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif len(attention_mask.size()) == 3:
                attention_mask = attention_mask.unsqueeze(1)
 
        static_embeddings, relative_embedding = self.embedding(input_ids.t())
        contextualized_embeddings, attention_probs = self.transformer(static_embeddings, attention_mask, relative_embedding)
        contextualized_embeddings = [e.transpose(0, 1) for e in contextualized_embeddings]
        last_layer = contextualized_embeddings[-1]
        contextualized_embeddings = [contextualized_embeddings[0]] + [
            contextualized_embeddings[i] - contextualized_embeddings[i - 1]
            for i in range(1, len(contextualized_embeddings))
        ]
        return last_layer, contextualized_embeddings, attention_probs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)

        if not return_dict:
            return (
                sequence_output,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


class LtgbertForMaskedLM(LtgbertModel):
    _keys_to_ignore_on_load_unexpected = ["head"]

    def __init__(self, config, **kwargs):
        super().__init__(config, add_mlm_layer=True, **kwargs)

    def get_output_embeddings(self):
        return self.classifier.nonlinearity[-1].weight

    def set_output_embeddings(self, new_embeddings):
        self.classifier.nonlinearity[-1].weight = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        subword_prediction = self.classifier(sequence_output)
        # subword_prediction[:, :, :16+1] = float("-inf")

        masked_lm_loss = None
        if labels is not None:
            labels_flatten = labels[:, 1:].flatten()
            subword_prediction_flatten = subword_prediction[:, :-1].flatten(0, 1)
            masked_lm_loss = F.cross_entropy(subword_prediction_flatten, labels_flatten)

        if not return_dict:
            output = (
                subword_prediction,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=subword_prediction,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


class Classifier(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()

        self.temperature = config.temperature
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = config.hidden_dropout_prob if drop_out is None else drop_out

        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(drop_out),
            nn.Linear(config.hidden_size, num_labels)
        )

    def forward(self, x):
        x = self.nonlinearity(x) / self.temperature
        return x


class LtgbertForCausalLM(LtgbertModel):
    _keys_to_ignore_on_load_unexpected = ["head"]

    def __init__(self, config, **kwargs):
        config.is_decoder = True
        super().__init__(config, add_mlm_layer=True, **kwargs)

    def get_output_embeddings(self):
        return self.classifier.nonlinearity[-1].weight

    def set_output_embeddings(self, new_embeddings):
        self.classifier.nonlinearity[-1].weight = new_embeddings

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer
    
    def can_generate(self):
        return True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutput]:

        assert inputs_embeds is None, "inputs_embeds is not supported for now"
        assert past_key_values is None, "past_key_values is not supported for now"
        assert not use_cache, "use_cache is not supported for now"
        # assert cache_position is None, "cache_position is not supported for now"

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        subword_prediction = self.classifier(sequence_output)
        # subword_prediction[:, :, :16+1] = float("-inf")

        masked_lm_loss = None
        if labels is not None:
            labels_flatten = labels[:, 1:].flatten()
            subword_prediction_flatten = subword_prediction[:, :-1].flatten(0, 1)
            masked_lm_loss = F.cross_entropy(subword_prediction_flatten, labels_flatten)

        if not return_dict:
            output = (
                subword_prediction,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return CausalLMOutput(
            loss=masked_lm_loss,
            logits=subword_prediction,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs



class LtgbertForSequenceClassification(LtgbertModel):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config, **kwargs):
        super().__init__(config, add_mlm_layer=False, **kwargs)

        self.num_labels = config.num_labels
        self.head = Classifier(config, self.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        logits = self.head(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (
                logits,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


class LtgbertForTokenClassification(LtgbertModel):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config, **kwargs):
        super().__init__(config, add_mlm_layer=False, **kwargs)

        self.num_labels = config.num_labels
        self.head = Classifier(config, self.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        logits = self.head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (
                logits,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


class LtgbertForQuestionAnswering(LtgbertModel):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config, **kwargs):
        super().__init__(config, add_mlm_layer=False, **kwargs)

        self.num_labels = config.num_labels
        self.head = Classifier(config, self.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(input_ids, attention_mask)
        logits = self.head(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


class LtgbertForMultipleChoice(LtgbertModel):
    _keys_to_ignore_on_load_unexpected = ["classifier"]
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config, **kwargs):
        super().__init__(config, add_mlm_layer=False, **kwargs)

        self.num_labels = getattr(config, "num_labels", 2)
        self.head = Classifier(config, self.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        sequence_output, contextualized_embeddings, attention_probs = self.get_contextualized_embeddings(flat_input_ids, flat_attention_mask)
        logits = self.head(sequence_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (
                reshaped_logits,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )
