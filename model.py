from diffusers import StableDiffusionPipeline

class Diffusion(StableDiffusionPipeline):
    def __init__(
        self, vae, text_encoder, tokenizer, unet, scheduler,
        safety_checker, feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler,
            safety_checker, feature_extractor, requires_safety_checker,
        )
