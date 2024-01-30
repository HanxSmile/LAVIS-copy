class DiffusionMixin:

    def _predict_noise(
            self,
            t,
            latent_model_input,
            text_embeddings,
            width=512,
            height=512,
            cond_image=None,
    ):
        if hasattr(self, "controlnet"):
            cond_image = prepare_cond_image(
                cond_image, width, height, batch_size=1, device=self.device
            )

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=cond_image,
                # conditioning_scale=controlnet_condition_scale,
                return_dict=False,
            )
        else:
            down_block_res_samples, mid_block_res_sample = None, None

        noise_pred = self.unet(
            latent_model_input,
            timestep=t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )["sample"]

        return noise_pred