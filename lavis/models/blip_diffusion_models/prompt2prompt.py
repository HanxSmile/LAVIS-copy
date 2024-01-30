from lavis.models.blip_diffusion_models.ptp_utils import (
    LocalBlend,
    P2PCrossAttnProcessor,
    AttentionRefine,
)


class Prompt2PromptMixin:

    def _register_attention_refine(
            self,
            src_subject,
            prompts,
            num_inference_steps,
            cross_replace_steps=0.8,
            self_replace_steps=0.4,
            threshold=0.3,
    ):
        device, tokenizer = self.device, self.tokenizer

        lb = LocalBlend(
            prompts=prompts,
            words=(src_subject,),
            device=device,
            tokenizer=tokenizer,
            threshold=threshold,
        )

        controller = AttentionRefine(
            prompts,
            num_inference_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            tokenizer=tokenizer,
            device=device,
            local_blend=lb,
        )

        self._register_attention_control(controller)

        return controller

    def _register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        if controller is not None:
            controller.num_att_layers = cross_att_count
