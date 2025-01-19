from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPVisionModel, CLIPVisionConfig, BaseModelOutputWithPooling
from transformers.models.llava_next import LlavaNextForConditionalGeneration, LlavaNextConfig, LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import LlavaNextMultiModalProjector, LlavaNextCausalLMOutputWithPast
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessorKwargs

from transformers.utils import logging
logger = logging.get_logger(__name__)

class ModifiedCLIPVisionTransformer(CLIPVisionTransformer):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        morphed_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)
        if morphed_mask is not None:
            morphed_mask = torch.cat([torch.Tensor([1]).to(device=morphed_mask.device), morphed_mask.flatten()])
            assert hidden_states.shape[0] == 1, f"{hidden_states.shape=}"
            hidden_states = hidden_states[:, morphed_mask == 1, :]
        # print(f"PRINTING SELECTED EMBEDDING")
        # for b in range(hidden_states.shape[0]):
        #     for i in range(hidden_states[b].shape[0]):
        #         print(f"{i=}, {list(map(float, list(hidden_states[b, i, :6])))=}")
        # print(f"DONE PRINTING SELECTED EMBEDDING")

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ModifiedCLIPVisionConfig(CLIPVisionConfig):
    model_type = 'modified_clip_vision_model'


class ModifiedCLIPVisionModel(CLIPVisionModel):
    config_class = ModifiedCLIPVisionConfig
    
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = ModifiedCLIPVisionTransformer(config)
        self.post_init()
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
        morphed_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
            morphed_mask=morphed_mask
        )

class ModifiedLlavaNextConfig(LlavaNextConfig):
    pass

from transformers import AutoModel, AutoConfig
AutoConfig.register('modified_clip_vision_model', ModifiedCLIPVisionConfig)
AutoModel.register(ModifiedCLIPVisionConfig, ModifiedCLIPVisionModel)

from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches

class ModifiedLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    def __init__(self, config: Union[LlavaNextConfig, ModifiedLlavaNextConfig]):
        if isinstance(config, LlavaNextConfig):
            config_dict = config.to_dict()
            config_dict['model_type'] = 'modified_clip_vision_model'
            config = ModifiedLlavaNextConfig(**config_dict)
        super().__init__(config)
        config.vision_config = ModifiedCLIPVisionConfig(**config.vision_config.to_dict())
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = LlavaNextMultiModalProjector(config)
        embed_std = 1 / math.sqrt(config.text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.text_config.hidden_size, dtype=self.dtype) * embed_std)

        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        morphed_mask: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        # print(f"{inputs_embeds=}")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # if the number of image tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            # In case we're in decoding stage, legacy behavior is checked by presence of pixel values even if use_cache=True
            if morphed_mask is None:
                legacy_processing = (
                    (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
                ) or (input_ids.shape[-1] == 1 and pixel_values is not None)
            else:
                legacy_processing = ((input_ids == self.config.image_token_index).sum(1).max() == 0) or (input_ids.shape[-1] == 1 and pixel_values is not None)

        image_features = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            if morphed_mask is None:
                image_features = self.get_image_features(
                    pixel_values,
                    image_sizes,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    morphed_mask=morphed_mask
                )

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
                image_features, feature_lens = self.pack_image_features(
                    image_features,
                    image_sizes,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_newline=self.image_newline,
                )
                if morphed_mask is not None:
                    image_features = image_features[morphed_mask.flatten() == 1, :]
                    feature_lens = torch.tensor([morphed_mask.flatten().sum()])
            else:
                image_features = [
                    self.get_image_features(
                        pv[torch.newaxis, :, :, :],
                        image_sizes,
                        vision_feature_layer=vision_feature_layer,
                        vision_feature_select_strategy=vision_feature_select_strategy,
                        morphed_mask=mm
                    )
                    for pv, mm in zip(pixel_values[0], morphed_mask)
                ]
                image_features = [x[0] for x in image_features]
                image_features = torch.cat([image_features[0][0]] + [self.image_newline[torch.newaxis, :]] + [im_feat[0] for im_feat in image_features[1:]])

        if legacy_processing:
            logger.warning_once(
                "MODEL Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
            )
            if input_ids.shape[1] != 1:
                raise Exception('bad case')
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, position_ids, labels, _ = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            else:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                # assert len(new_batch_index) == 0, f"{new_batch_index=}"
                # assert len(new_non_attended_tokens) == 0
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[-target_length:]

        # TODO: @raushan retain only the new behavior after v4.47
        elif image_features is not None:
            n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            n_image_features = image_features.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
    
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int,
        vision_feature_select_strategy: str,
        morphed_mask: Optional[torch.Tensor] = None,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_layer (`int`):
                The index of the layer to select the vision feature.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (List[`torch.Tensor`]): List of image feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_patches, image_length, embed_dim)`).
        """
        from torchvision import transforms
        # ! infer image_num_patches from image_sizes
        if morphed_mask is None:
            image_num_patches = [
                image_size_to_num_patches(
                    image_size=imsize,
                    grid_pinpoints=self.config.image_grid_pinpoints,
                    patch_size=self.config.vision_config.image_size,
                )
                for imsize in image_sizes
            ]
        else:
            image_num_patches = [1]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True, morphed_mask=morphed_mask)
        selected_image_feature = image_features.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        return image_features

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes
            if 'morphed_mask' in kwargs:
                model_inputs['morphed_mask'] = kwargs['morphed_mask']

        return model_inputs


from transformers.models.llava_next.processing_llava_next import (
    ImageInput, TextInput, PreTokenizedInput, BatchFeature, Unpack,
    _validate_images_text_input_order, get_image_size, to_numpy_array
)

class ModifiedLlavaNextProcessor(LlavaNextProcessor):
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[LlavaNextProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        num_viz_tokens = None
        if 'num_viz_tokens' in kwargs:
            num_viz_tokens = kwargs['num_viz_tokens']
            del kwargs['num_viz_tokens']

        output_kwargs = self._merge_kwargs(
            LlavaNextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = text
        if image_inputs:
            if self.patch_size is None or self.vision_feature_select_strategy is None:
                logger.warning_once(
                    "PROCESSOR Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. "
                    "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                    "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                    "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                )
            else:
                image_sizes = iter(image_inputs["image_sizes"])
                height, width = get_image_size(to_numpy_array(image_inputs["pixel_values"][0][0]))
                prompt_strings = []
                for sample in text:
                    while self.image_token in sample:
                        image_size = next(image_sizes)
                        orig_height, orig_width = image_size
                        if num_viz_tokens is not None:
                            num_image_tokens = num_viz_tokens
                        else:
                            num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
                            if self.vision_feature_select_strategy == "default":
                                num_image_tokens -= 1
                        sample = sample.replace(self.image_token, "<placeholder>" * num_image_tokens, 1)
                    prompt_strings.append(sample)
                prompt_strings = [sample.replace("<placeholder>", self.image_token) for sample in prompt_strings]

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})
