import torch
import torch.nn as nn
from lavis.models.blip_diffusion_models.utils import numpy_to_pil


class ProcessMixin:

    def _inversion_transform(self, image, target_size=512):
        from torchvision import transforms

        tform = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
            ]
        )
        image = tform(image).unsqueeze(0).to(self.device)
        return 2.0 * image - 1.0

    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        return rv

    def _build_prompts_edit(self, cond_subject, tgt_subject, prompt):
        placeholder = " ".join(["sks"] * self.num_query_token)

        src_prompt = f"a {cond_subject} {prompt}"
        tgt_prompt = f"a {placeholder} {tgt_subject} {prompt}"

        return [src_prompt, tgt_prompt]

    def before_training(self, dataset, **kwargs):
        assert len(dataset) == 1, "Only support single dataset for now."

        key = list(dataset.keys())[0]
        dataset = dataset[key]["train"]

        # collect all examples
        # [FIXME] this is not memory efficient. may OOM if the dataset is large.
        examples = [dataset[i] for i in range(dataset.len_without_repeat)]
        input_images = (
            torch.stack([example["inp_image"] for example in examples])
            .to(memory_format=torch.contiguous_format)
            .float()
        ).to(self.device)
        subject_text = [dataset.subject for _ in range(input_images.shape[0])]

        # calculate ctx embeddings and cache them
        ctx_embeddings = self.forward_ctx_embeddings(
            input_image=input_images, text_input=subject_text
        )
        # take mean of all ctx embeddings
        ctx_embeddings = ctx_embeddings.mean(dim=0, keepdim=True)
        self.ctx_embeddings_cache = nn.Parameter(ctx_embeddings, requires_grad=True)
        self._use_embeddings_cache = True

        # free up CUDA memory
        self.blip.to("cpu")
        self.proj_layer.to("cpu")

        torch.cuda.empty_cache()

    def _init_latent(self, latent, height, width, generator, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                device=generator.device,
            )
        latent = latent.expand(
            batch_size,
            self.unet.in_channels,
            height // 8,
            width // 8,
        )
        return latent.to(self.device)

    @torch.no_grad()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        assert isinstance(image, torch.Tensor)

        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    def _latent_to_image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = numpy_to_pil(image)

        return image

    def _tokenize_text(self, text_input, with_query=True):
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        if with_query:
            max_len -= self.num_query_token

        tokenized_text = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        return tokenized_text
